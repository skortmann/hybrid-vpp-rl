"""Run baseline controllers over a date range and report KPIs.

Edit the CONFIG block and run::

    uv run python -m hybrid_vpp.evaluation.run_baselines
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from hybrid_vpp.config.models import ExperimentConfig, load_config
from hybrid_vpp.controllers.base import run_episode
from hybrid_vpp.controllers.rule_based import RuleBasedController
from hybrid_vpp.controllers.simple import DoNothingController
from hybrid_vpp.data.site_profiles import load_site_profiles
from hybrid_vpp.data.sqlite_market_data import MarketDataStore
from hybrid_vpp.evaluation.metrics import episode_metrics, metrics_frame
from hybrid_vpp.forecasts.price import build_price_forecaster
from hybrid_vpp.forecasts.renewable import build_renewable_forecaster
from hybrid_vpp.sim.simulator import Simulator

log = logging.getLogger(__name__)


def build_stack(cfg: ExperimentConfig):
    """Shared data/simulator/forecaster stack for controllers and envs."""
    store = MarketDataStore(cfg.data, cfg.markets, cfg.synthetic_market)
    profiles = load_site_profiles(cfg.data, cfg.site, store)
    sim = Simulator(cfg, store, profiles)
    renewable_fc = build_renewable_forecaster(
        cfg.forecast.renewable_mode,
        profiles,
        store.zone_renewables(),
        cfg.site,
        sigma=cfg.forecast.noisy_oracle_sigma,
    )
    price_fc = build_price_forecaster(
        cfg.forecast.price_mode,
        sim.calendar,
        {
            "daa": store.daa_prices()["price_eur_per_mwh"],
            "ida1": store.ida_prices("ida1"),
            "ida2": store.ida_prices("ida2"),
            "ida3": store.ida_prices("ida3"),
            "idc": store.idc_indices()["IDFULL"],
        },
    )
    return store, profiles, sim, renewable_fc, price_fc


def evaluate_controller(
    name: str,
    controller,
    sim: Simulator,
    days: pd.DatetimeIndex,
) -> pd.DataFrame:
    per_day: dict[str, dict[str, float]] = {}
    for day in days:
        try:
            run_episode(sim, controller, day)
        except Exception:
            log.exception("%s failed on %s", name, day.date())
            continue
        per_day[str(day.date())] = episode_metrics(sim)
    return metrics_frame(per_day)


# --------------------------------------------------------------------------
# CONFIG — edit and run as a module (no CLI flags by design)
# --------------------------------------------------------------------------
CONFIG_PATH = Path("configs/default.yaml")
START_DAY = "2025-03-10"
N_DAYS = 14
CONTROLLERS = ("rule_based", "do_nothing")
SUMMARY_COLUMNS = [
    "total_net_revenue_eur",
    "market_revenue_eur",
    "imbalance_cash_eur",
    "abs_deviation_mwh",
    "wind_curtailed_mwh",
    "pv_curtailed_mwh",
    "bess_charged_mwh",
    "equivalent_full_cycles",
    "corrected_dispatch_intervals",
]

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(CONFIG_PATH)
    store, profiles, sim, renewable_fc, price_fc = build_stack(cfg)
    days = pd.date_range(START_DAY, periods=N_DAYS, freq="D")

    controllers = {
        "rule_based": RuleBasedController(cfg, renewable_fc, price_fc),
        "do_nothing": DoNothingController(cfg, renewable_fc),
    }
    for name in CONTROLLERS:
        df = evaluate_controller(name, controllers[name], sim, days)
        print(f"\n=== {name} ({days[0].date()} .. {days[-1].date()}) ===")
        print(df.loc["TOTAL", SUMMARY_COLUMNS].to_string())
