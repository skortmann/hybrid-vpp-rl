"""Baseline controllers: full-episode runs with accounting verification."""

from pathlib import Path

import pandas as pd
import pytest

from hybrid_vpp.config.models import load_config
from hybrid_vpp.controllers.base import run_episode
from hybrid_vpp.controllers.rule_based import RuleBasedController
from hybrid_vpp.controllers.simple import DoNothingController
from hybrid_vpp.evaluation.metrics import episode_metrics
from hybrid_vpp.evaluation.run_baselines import build_stack
from tests.conftest import requires_real_db

CONFIG = Path(__file__).parents[2] / "configs" / "default.yaml"

pytestmark = requires_real_db


@pytest.fixture(scope="module")
def stack():
    cfg = load_config(CONFIG)
    return cfg, *build_stack(cfg)


def test_rule_based_episode_accounting(stack):
    cfg, store, profiles, sim, renewable_fc, price_fc = stack
    controller = RuleBasedController(cfg, renewable_fc, price_fc)
    run_episode(sim, controller, "2025-03-15")

    metrics = episode_metrics(sim)
    # every euro exactly once
    assert sim.ledger.total() == pytest.approx(sum(sim.ledger.by_component().values()))
    assert metrics["total_net_revenue_eur"] == pytest.approx(sim.ledger.total())
    # controller trades and uses the battery
    assert metrics["daa_volume_mwh"] > 0
    assert metrics["bess_charged_mwh"] > 0
    # grid limit never violated
    for r in sim.dispatch_records.values():
        assert r.dispatch.grid_power_mw <= cfg.site.grid.export_limit_mw + 1e-6
    # energy balance per interval
    for r in sim.dispatch_records.values():
        assert r.dispatch.grid_power_mw == pytest.approx(
            r.wind_avail_mw
            - r.dispatch.wind_curtail_mw
            + r.pv_avail_mw
            - r.dispatch.pv_curtail_mw
            + r.dispatch.bess_power_mw,
            abs=1e-9,
        )


def test_do_nothing_never_uses_battery(stack):
    cfg, store, profiles, sim, renewable_fc, price_fc = stack
    controller = DoNothingController(cfg, renewable_fc)
    run_episode(sim, controller, "2025-03-16")
    metrics = episode_metrics(sim)
    assert metrics["bess_charged_mwh"] == 0.0
    assert metrics["bess_discharged_mwh"] == 0.0
    assert metrics["idc_volume_mwh"] == 0.0
    assert metrics["daa_volume_mwh"] > 0


def test_rule_based_deterministic(stack):
    cfg, store, profiles, sim, renewable_fc, price_fc = stack
    controller = RuleBasedController(cfg, renewable_fc, price_fc)
    run_episode(sim, controller, "2025-03-17")
    first = sim.ledger.total()
    run_episode(sim, controller, "2025-03-17")
    assert sim.ledger.total() == pytest.approx(first)


def test_multi_day_episode(stack):
    cfg, store, profiles, sim, renewable_fc, price_fc = stack
    controller = RuleBasedController(cfg, renewable_fc, price_fc)
    run_episode(sim, controller, "2025-03-18", days=2)
    assert len(sim.dispatch_records) == 192
    assert len(sim.settlements) == 192
    starts = sorted(sim.dispatch_records)
    diffs = {b - a for a, b in zip(starts, starts[1:], strict=False)}
    assert diffs == {pd.Timedelta(minutes=15)}
