"""Evaluate a trained RL policy against all baselines on a chosen split.

For every valid day of the split, runs deterministic episodes with:

* the trained RL policy (SB3 checkpoint),
* the do-nothing baseline,
* the rule-based baseline,
* the rolling-horizon MILP benchmark (same forecasts as the RL agent),
* the perfect-foresight MILP upper bound.

Writes one CSV per controller plus a summary table. Edit CONFIG and run::

    uv run python -m hybrid_vpp.training.evaluate
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from hybrid_vpp.config.models import ExperimentConfig, load_config
from hybrid_vpp.controllers.base import run_episode
from hybrid_vpp.controllers.optimization import OptimizationController
from hybrid_vpp.controllers.rule_based import RuleBasedController
from hybrid_vpp.controllers.simple import DoNothingController
from hybrid_vpp.evaluation.metrics import episode_metrics, metrics_frame
from hybrid_vpp.evaluation.run_baselines import build_stack
from hybrid_vpp.forecasts.price import PerfectPriceForecast
from hybrid_vpp.forecasts.renewable import PerfectForesightForecast

log = logging.getLogger(__name__)


def evaluate_rl_policy(
    model_path: Path, cfg: ExperimentConfig, split: str, days: pd.DatetimeIndex
) -> pd.DataFrame:
    from stable_baselines3 import PPO, SAC

    from hybrid_vpp.envs.hybrid_vpp_env import HybridVppEnv

    algo_cls = {"ppo": PPO, "sac": SAC}[cfg.training.algorithm]
    model = algo_cls.load(model_path)
    env = HybridVppEnv(cfg, split=split)
    per_day = {}
    for day in days:
        obs, info = env.reset(options={"day": day})
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, info = env.step(action)
        per_day[str(day.date())] = info["episode_metrics"]
    return metrics_frame(per_day)


def evaluate_baselines(
    cfg: ExperimentConfig, days: pd.DatetimeIndex
) -> tuple[dict[str, pd.DataFrame], dict]:
    store, profiles, sim, renewable_fc, price_fc = build_stack(cfg)
    provenance = {"path": str(store.db_path), "provenance": store.provenance}
    perfect_prices = PerfectPriceForecast(
        {
            "daa": store.daa_prices()["price_eur_per_mwh"],
            "ida1": store.ida_prices("ida1"),
            "ida2": store.ida_prices("ida2"),
            "ida3": store.ida_prices("ida3"),
            "idc": store.idc_indices()["IDFULL"],
        }
    )
    controllers = {
        "do_nothing": DoNothingController(cfg, renewable_fc),
        "rule_based": RuleBasedController(cfg, renewable_fc, price_fc),
        "milp": OptimizationController(cfg, renewable_fc, price_fc),
        "milp_perfect_foresight": OptimizationController(
            cfg, PerfectForesightForecast(profiles), perfect_prices
        ),
    }
    out = {}
    for name, controller in controllers.items():
        per_day = {}
        for day in days:
            try:
                run_episode(sim, controller, day)
                per_day[str(day.date())] = episode_metrics(sim)
            except Exception:
                log.exception("%s failed on %s", name, day.date())
        out[name] = metrics_frame(per_day)
        log.info(
            "%s: total %.0f EUR over %d days",
            name,
            out[name].loc["TOTAL", "total_net_revenue_eur"],
            len(per_day),
        )
    return out, provenance


def evaluate(
    config_path: Path,
    model_path: Path | None,
    split: str,
    out_dir: Path,
    max_days: int | None = None,
) -> pd.DataFrame:
    from hybrid_vpp.envs.hybrid_vpp_env import HybridVppEnv

    cfg = load_config(config_path)
    env = HybridVppEnv(cfg, split=split)  # reuse valid-day logic
    days = env.valid_days
    if max_days:
        days = days[:max_days]
    del env

    results, provenance = evaluate_baselines(cfg, days)
    if model_path is not None:
        results["rl"] = evaluate_rl_policy(model_path, cfg, split, days)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = {}
    for name, df in results.items():
        df.to_csv(out_dir / f"evaluation_{split}_{name}.csv")
        summary_rows[name] = df.loc["TOTAL"]
    summary = pd.DataFrame(summary_rows).T
    summary.to_csv(out_dir / f"evaluation_{split}_summary.csv")
    (out_dir / f"evaluation_{split}_metadata.json").write_text(
        json.dumps(
            {
                "market_data": provenance,
                "days": [str(d.date()) for d in days],
                "model": str(model_path) if model_path else None,
            },
            indent=2,
        )
    )
    print(f"market data: {provenance['path']} (provenance: {provenance['provenance']})")

    cols = [
        "total_net_revenue_eur",
        "market_revenue_eur",
        "imbalance_cash_eur",
        "abs_deviation_mwh",
        "equivalent_full_cycles",
        "congestion_wind_curtailed_mwh",
        "congestion_pv_curtailed_mwh",
    ]
    print(f"\n=== {split} summary ({len(days)} days) ===")
    print(summary[cols].round(0).to_string())
    return summary


# --------------------------------------------------------------------------
# CONFIG — edit and run as a module (no CLI flags by design)
# --------------------------------------------------------------------------
CONFIG_PATH = Path("configs/default.yaml")
MODEL_PATH: Path | None = None  # e.g. Path("runs/ppo-baseline_.../best/best_model.zip")
SPLIT = "val"
OUT_DIR = Path("runs/evaluation")
MAX_DAYS: int | None = None

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    evaluate(CONFIG_PATH, MODEL_PATH, SPLIT, OUT_DIR, MAX_DAYS)
