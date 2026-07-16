"""Controlled experiment runner and registry for the RL research program.

An experiment = (config overrides, algorithm, action mode, seed, budget).
Every run records provenance (git commit, schema versions, W&B id), trains
with the standard entry point, evaluates the best validation checkpoint on
a FIXED set of validation days (identical across all experiments), and
appends one JSON line to ``experiments/registry.jsonl``.

Screening evaluations use the first ``SCREEN_DAYS`` validation days;
finalists are re-evaluated on the full validation split. The test split is
never touched here.

Edit the CONFIG block and run::

    uv run python -m hybrid_vpp.training.experiments
"""

from __future__ import annotations

import json
import logging
import subprocess
import time as _time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from hybrid_vpp.config.models import ExperimentConfig, load_config

log = logging.getLogger(__name__)

REGISTRY = Path("experiments/registry.jsonl")
SCREEN_DAYS = 30


@dataclass
class ExperimentSpec:
    experiment_id: str
    phase: str  # smoke | learnability | screening | tuning | confirmation
    algorithm: str = "ppo"
    action_mode: str = "direct"
    total_timesteps: int = 150_000
    seed: int = 0
    n_envs: int = 8
    #: dot-path config overrides, e.g. {"markets.idc.enabled": False}
    overrides: dict[str, Any] = field(default_factory=dict)
    algo_kwargs: dict[str, Any] = field(default_factory=dict)
    notes: str = ""


def _apply_overrides(raw: dict, overrides: dict[str, Any]) -> dict:
    for path, value in overrides.items():
        node = raw
        keys = path.split(".")
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = value
    return raw


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        ).stdout.strip()
    except OSError:
        return "unknown"


def build_config(spec: ExperimentSpec, base_config: Path) -> tuple[ExperimentConfig, Path]:
    with open(base_config) as fh:
        raw = yaml.safe_load(fh)
    raw = _apply_overrides(raw, spec.overrides)
    training = raw.setdefault("training", {})
    training.update(
        algorithm=spec.algorithm,
        total_timesteps=spec.total_timesteps,
        seed=spec.seed,
        n_envs=spec.n_envs,
        run_name=spec.experiment_id,
        algo_kwargs=spec.algo_kwargs,
    )
    raw.setdefault("episode", {})["action_mode"] = spec.action_mode
    out_dir = Path("runs/experiments")
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / f"{spec.experiment_id}.yaml"
    config_path.write_text(yaml.safe_dump(raw))
    return load_config(config_path), config_path


def evaluate_checkpoint(
    config_path: Path, model_path: Path, days: pd.DatetimeIndex
) -> dict[str, float]:
    """Deterministic evaluation of a checkpoint on fixed days (val split)."""
    from hybrid_vpp.envs.hybrid_vpp_env import HybridVppEnv
    from hybrid_vpp.training.algorithms import algo_class

    cfg = load_config(config_path)
    model = algo_class(cfg.training.algorithm).load(model_path)
    env = HybridVppEnv(cfg, split="val")
    revenues, corrections = [], []
    for day in days:
        obs, info = env.reset(options={"day": day})
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, info = env.step(action)
        m = info["episode_metrics"]
        revenues.append(m["total_net_revenue_eur"])
        corrections.append(m["corrected_dispatch_intervals"])
    revenue = np.asarray(revenues)
    return {
        "days": len(days),
        "mean_revenue_eur": float(revenue.mean()),
        "median_revenue_eur": float(np.median(revenue)),
        "std_revenue_eur": float(revenue.std()),
        "total_revenue_eur": float(revenue.sum()),
        "mean_corrected_intervals": float(np.mean(corrections)),
        "per_day_revenue_eur": [float(v) for v in revenue],
    }


def evaluation_days(config_path: Path, n_days: int | None = SCREEN_DAYS) -> pd.DatetimeIndex:
    from hybrid_vpp.envs.hybrid_vpp_env import HybridVppEnv

    env = HybridVppEnv(load_config(config_path), split="val")
    days = env.valid_days
    return days[:n_days] if n_days else days


def run_experiment(spec: ExperimentSpec, base_config: Path = Path("configs/default.yaml")) -> dict:
    from hybrid_vpp.envs.actions import ACTION_SCHEMA_VERSIONS
    from hybrid_vpp.training.train import train

    cfg, config_path = build_config(spec, base_config)
    started = _time.time()
    run_dir = train(config_path)
    wall_clock_s = _time.time() - started

    best = run_dir / "best" / "best_model.zip"
    model_path = best if best.exists() else run_dir / "final_model.zip"
    days = evaluation_days(config_path)
    evaluation = evaluate_checkpoint(config_path, model_path, days)

    record = {
        "experiment_id": spec.experiment_id,
        "phase": spec.phase,
        "timestamp": datetime.now().astimezone().isoformat(),
        "git_commit": _git_commit(),
        "environment_version": "env-v1",
        "action_schema": ACTION_SCHEMA_VERSIONS[spec.action_mode],
        "algorithm": spec.algorithm,
        "action_mode": spec.action_mode,
        "seed": spec.seed,
        "total_timesteps": spec.total_timesteps,
        "wall_clock_s": round(wall_clock_s, 1),
        "overrides": spec.overrides,
        "algo_kwargs": spec.algo_kwargs,
        "run_dir": str(run_dir),
        "model_path": str(model_path),
        "config_path": str(config_path),
        "validation": evaluation,
        "notes": spec.notes,
    }
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    with REGISTRY.open("a") as fh:
        fh.write(json.dumps(record) + "\n")
    log.info(
        "experiment %s: mean %.0f EUR/day over %d val days (%.0f min)",
        spec.experiment_id,
        evaluation["mean_revenue_eur"],
        evaluation["days"],
        wall_clock_s / 60,
    )
    return record


def run_many(specs: list[ExperimentSpec], base_config: Path = Path("configs/default.yaml")):
    results = []
    for spec in specs:
        try:
            results.append(run_experiment(spec, base_config))
        except Exception:
            log.exception("experiment %s failed", spec.experiment_id)
    return results


# --------------------------------------------------------------------------
# CONFIG — experiment matrix; edit RUN_ONLY to select, then run as a module
# --------------------------------------------------------------------------
#: Level-1 learnability: DAA + battery only, perfect forecasts (deterministic)
LEVEL1_OVERRIDES = {
    "markets.ida1.enabled": False,
    "markets.ida2.enabled": False,
    "markets.ida3.enabled": False,
    "markets.idc.enabled": False,
    "forecast.renewable_mode": "perfect",
    "forecast.price_mode": "perfect",
    "training.tracker": "wandb",
}

SPECS: list[ExperimentSpec] = [
    # ---- Phase 1: learnability on the deterministic DAA-only level
    ExperimentSpec(
        "L1-ppo-hourly-target",
        "learnability",
        algorithm="ppo",
        action_mode="hourly_target",
        total_timesteps=200_000,
        overrides=LEVEL1_OVERRIDES,
        notes="Level-1: PPO must approach the perfect-forecast MILP optimum",
    ),
    ExperimentSpec(
        "L1-sac-hourly-target",
        "learnability",
        algorithm="sac",
        action_mode="hourly_target",
        total_timesteps=150_000,
        n_envs=4,
        overrides=LEVEL1_OVERRIDES,
        notes="Level-1 off-policy comparison",
    ),
    # ---- Phase 2: action-space screening on the full environment
    ExperimentSpec(
        "S2-ppo-direct",
        "screening",
        algorithm="ppo",
        action_mode="direct",
        total_timesteps=300_000,
        notes="reference: current 103-dim direct action",
    ),
    ExperimentSpec(
        "S2-ppo-target",
        "screening",
        algorithm="ppo",
        action_mode="target_position",
        total_timesteps=300_000,
    ),
    ExperimentSpec(
        "S2-ppo-hourly",
        "screening",
        algorithm="ppo",
        action_mode="hourly_target",
        total_timesteps=300_000,
    ),
    ExperimentSpec(
        "S2-ppo-residual",
        "screening",
        algorithm="ppo",
        action_mode="residual_hourly",
        total_timesteps=300_000,
    ),
    ExperimentSpec(
        "S2-sac-hourly",
        "screening",
        algorithm="sac",
        action_mode="hourly_target",
        total_timesteps=200_000,
        n_envs=4,
    ),
    ExperimentSpec(
        "S2-tqc-hourly",
        "screening",
        algorithm="tqc",
        action_mode="hourly_target",
        total_timesteps=200_000,
        n_envs=4,
    ),
    ExperimentSpec(
        "S2-tqc-residual",
        "screening",
        algorithm="tqc",
        action_mode="residual_hourly",
        total_timesteps=200_000,
        n_envs=4,
    ),
    # ---- Tier 1 (advanced program): strategic actions and CrossQ
    ExperimentSpec(
        "S3-sac-strategic",
        "screening",
        algorithm="sac",
        action_mode="strategic",
        total_timesteps=120_000,
        n_envs=4,
        notes="Tier-1: 7-dim strategic actions, SAC",
    ),
    ExperimentSpec(
        "S3-crossq-strategic",
        "screening",
        algorithm="crossq",
        action_mode="strategic",
        total_timesteps=120_000,
        n_envs=4,
        notes="Tier-1: CrossQ (SBX) on strategic actions",
    ),
    ExperimentSpec(
        "S3-crossq-hourly",
        "screening",
        algorithm="crossq",
        action_mode="hourly_target",
        total_timesteps=200_000,
        n_envs=4,
        notes="Tier-1: CrossQ on hourly target-position actions",
    ),
]

RUN_ONLY: list[str] | None = None  # e.g. ["L1-ppo-hourly-target"]
BASE_CONFIG = Path("configs/default.yaml")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    selected = [s for s in SPECS if RUN_ONLY is None or s.experiment_id in RUN_ONLY]
    run_many(selected, BASE_CONFIG)
