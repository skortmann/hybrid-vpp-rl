"""Chained carry-over evaluation over a contiguous horizon.

Replays consecutive delivery days with the battery state carried from each
day into the next (physically faithful operation, no daily SoC reset). Days
missing from the split's valid set break the chain: the battery restarts at
``soc_initial`` there, with a note — carry-over across unsimulated gaps is
not physically defined. Reports per-day revenue under both valuations and
horizon totals; under carry-over the per-day ``terminal_energy_value_eur``
prices real inventory changes, so the adjusted totals telescope cleanly.

Run as ``uv run python -m hybrid_vpp.evaluation.carry_over_eval``; edit the
CONFIG block or call :func:`chained_horizon_eval` with keyword arguments.
``experiment_id=None`` evaluates the rule-based controller; otherwise the
checkpoint is resolved from ``experiments/registry.jsonl``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from hybrid_vpp.config.models import load_config

# --------------------------------------------------------------------- CONFIG

EXPERIMENT_ID: str | None = "V6-hybrid-sac-strategic-seed0"  # None -> rule-based
CONFIG_PATH = Path("configs/default.yaml")
REGISTRY = Path("experiments/registry.jsonl")
SPLIT = "val"
START_DAY = "2025-11-01"
N_DAYS = 30

# ------------------------------------------------------------------------


def _registry_checkpoint(experiment_id: str) -> tuple[Path, Path]:
    found = None
    for line in REGISTRY.read_text().splitlines():
        record = json.loads(line)
        if record.get("experiment_id") == experiment_id:
            found = (Path(record["model_path"]), Path(record["config_path"]))
    if found is None:
        raise LookupError(f"experiment {experiment_id!r} not found in {REGISTRY}")
    return found


def chained_horizon_eval(
    experiment_id: str | None = EXPERIMENT_ID,
    config_path: Path = CONFIG_PATH,
    split: str = SPLIT,
    start_day: str = START_DAY,
    n_days: int = N_DAYS,
) -> pd.DataFrame:
    """Carry-over replay; returns the per-day metrics frame (printed too)."""
    from hybrid_vpp.envs.hybrid_vpp_env import HybridVppEnv

    model = None
    if experiment_id is not None:
        from hybrid_vpp.training.algorithms import algo_class

        model_path, config_path = _registry_checkpoint(experiment_id)
        cfg = load_config(config_path)
        model = algo_class(cfg.training.algorithm).load(model_path)
    else:
        cfg = load_config(config_path)

    env = HybridVppEnv(cfg, split=split)
    controller = None
    if model is None:
        from hybrid_vpp.controllers.rule_based import RuleBasedController

        controller = RuleBasedController(
            cfg, env.obs_builder.renewable_forecaster, env.obs_builder.price_forecaster
        )
    valid = set(env.valid_days.normalize())

    carried_soc: float | None = None
    rows = []
    for day in pd.date_range(start_day, periods=n_days, freq="D"):
        if day not in valid:
            print(f"  {day.date()}: not simulable in split {split!r} — chain broken, SoC resets")
            carried_soc = None
            continue
        if model is not None:
            options: dict = {"day": day}
            if carried_soc is not None:
                options["initial_soc"] = carried_soc
            obs, info = env.reset(options=options)
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, _, info = env.step(action)
            m = info["episode_metrics"]
        else:
            from hybrid_vpp.controllers.base import run_episode
            from hybrid_vpp.evaluation.metrics import episode_metrics

            run_episode(env.sim, controller, day, initial_soc=carried_soc)
            m = episode_metrics(env.sim)
        rows.append(
            {
                "day": str(day.date()),
                "start_soc": env.sim.episode_start_energy_mwh
                / cfg.site.battery.energy_capacity_mwh,
                "end_soc": m["final_soc"],
                "revenue_eur": m["total_net_revenue_eur"],
                "terminal_value_eur": m["terminal_energy_value_eur"],
                "revenue_adjusted_eur": m["total_net_revenue_terminal_adjusted_eur"],
            }
        )
        carried_soc = float(m["final_soc"])
    frame = pd.DataFrame(rows).set_index("day")
    print(frame.round(2).to_string())
    print(
        f"horizon totals: ledger {frame['revenue_eur'].sum():,.0f} €, "
        f"adjusted {frame['revenue_adjusted_eur'].sum():,.0f} €, "
        f"mean start SoC {frame['start_soc'].mean():.2f}"
    )
    return frame


if __name__ == "__main__":
    chained_horizon_eval()
