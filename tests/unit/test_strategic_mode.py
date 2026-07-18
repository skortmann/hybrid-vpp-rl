"""Strategic action mode (act-v5): translator semantics and env integration."""

from pathlib import Path

import numpy as np
import pytest

from hybrid_vpp.config.models import load_config
from hybrid_vpp.envs.hybrid_vpp_env import HybridVppEnv

CONFIG = Path(__file__).parents[2] / "configs" / "synthetic_market.yaml"


@pytest.fixture(scope="module")
def env() -> HybridVppEnv:
    cfg = load_config(CONFIG)
    cfg.episode.action_mode = "strategic"
    return HybridVppEnv(cfg, split="train")


def test_action_space_is_seven_dimensional(env):
    assert env.action_space.shape == (7,)
    # observations keep the hourly per-slot arrays
    assert env.observation_space.shape[0] == 23 + 5 * 25


def test_event_masks(env):
    obs, info = env.reset(seed=0, options={"day": "2025-05-12"})
    assert info["event_type"] == "DAA_GATE_CLOSURE"
    assert info["action_mask"].tolist() == [1, 1, 0, 0, 0, 0, 0]


def test_midrange_action_reproduces_rule_based(env):
    """coverage=1, gains=1, threshold=0, bias=0 == the rule-based controller."""
    # raw values: coverage 1.0 -> a0 = 2/1.2 - 1; scale/gain 1.0 -> a = 1
    a = np.array([2 / 1.2 - 1, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    obs, _ = env.reset(seed=0, options={"day": "2025-05-14"})
    done = False
    while not done:
        obs, r, done, _, info = env.step(a)
    strategic_total = env.sim.ledger.total()

    from hybrid_vpp.controllers.base import run_episode
    from hybrid_vpp.controllers.rule_based import RuleBasedController

    controller = RuleBasedController(
        env.cfg,
        env.obs_builder.renewable_forecaster,
        env.obs_builder.price_forecaster,
    )
    run_episode(env.sim, controller, "2025-05-14")
    rb_total = env.sim.ledger.total()
    assert strategic_total == pytest.approx(rb_total, rel=1e-6)


def test_zero_coverage_sells_nothing_day_ahead(env):
    a = np.full(7, -1.0, dtype=np.float32)  # coverage 0, no arbitrage
    env.reset(seed=0, options={"day": "2025-05-15"})
    env.step(a)  # DAA gate
    assert env.sim.book.turnover_mwh("daa") == pytest.approx(0.0)


def test_gain_scales_ida_orders(env):
    full = np.array([2 / 1.2 - 1, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    env.reset(seed=0, options={"day": "2025-05-16"})
    env.step(full)  # DAA
    trades_before = len(env.sim.book.trades)
    zero_gain = full.copy()
    zero_gain[2] = -1.0  # ida_gain = 0
    env.step(zero_gain)  # IDA1: no corrections traded
    assert len(env.sim.book.trades) == trades_before


def test_full_episode_random_strategic(env):
    obs, _ = env.reset(seed=3, options={"day": "2025-05-20"})
    done = False
    while not done:
        obs, r, done, _, info = env.step(env.action_space.sample())
        assert np.isfinite(r)
    m = info["episode_metrics"]
    assert m["daa_volume_mwh"] >= 0
    # accounting invariant holds under the translator
    assert env.sim.ledger.total() == pytest.approx(sum(env.sim.ledger.by_component().values()))
