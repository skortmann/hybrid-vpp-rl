"""Isolated experiment worker: claim one job, run it, validate, exit.

A worker is a *fresh* Python process (spawned by the supervisor) that
atomically claims the highest-priority QUEUED job from the research
registry, executes it, validates the result, marks the terminal state,
and exits — one experiment per process, never two. PyTorch/JAX/W&B and
all vectorized environments are initialized only inside this process
(and its spawned children).

Job kinds:

* ``training`` — spec_json is an ExperimentSpec dict; runs training with
  heartbeats, evaluates the best checkpoint on the fixed validation days,
  revalidates (checkpoint reload + finite values + aggregate recompute),
  writes the ``DONE`` marker atomically.
* ``milp_anchor_l1`` — computes the exact Level-1 MILP anchor.

W&B failures never kill a valid run: on any wandb error the worker
retries the whole experiment once in offline mode (WANDB_MODE=offline).

Run as a module (claims one job, then exits)::

    uv run python -m hybrid_vpp.training.worker
"""

from __future__ import annotations

import json
import logging
import os
import traceback
from pathlib import Path

import numpy as np

from hybrid_vpp.training.research_state import Heartbeat, ResearchRegistry, utcnow

log = logging.getLogger(__name__)

REGISTRY_PATH = Path("runs/research_state.sqlite")


def _write_done(run_dir: Path, payload: dict) -> None:
    tmp = run_dir / "DONE.tmp"
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(run_dir / "DONE")


def validate_training_result(
    result: dict, config_path: Path, model_path: Path, expected_days: int = 30
) -> list[str]:
    """Independent validation of a completed training result."""
    problems: list[str] = []
    per_day = result.get("per_day_revenue_eur", [])
    if len(per_day) != expected_days:
        problems.append(f"expected {expected_days} evaluation days, got {len(per_day)}")
    if not all(np.isfinite(v) for v in per_day):
        problems.append("non-finite per-day revenue values")
    if per_day:
        recomputed = float(np.mean(per_day))
        if abs(recomputed - result["mean_revenue_eur"]) > 1e-6 * max(1, abs(recomputed)):
            problems.append("aggregate mean does not equal recomputed episode mean")
    if not Path(model_path).exists() or Path(model_path).stat().st_size < 1000:
        problems.append(f"checkpoint missing or implausibly small: {model_path}")
    else:
        try:  # reload in this process to prove the artifact is usable
            from hybrid_vpp.config.models import load_config
            from hybrid_vpp.training.algorithms import algo_class

            cfg = load_config(config_path)
            algo_class(cfg.training.algorithm).load(model_path)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"checkpoint reload failed: {exc}")
    return problems


def run_training_job(registry: ResearchRegistry, row: dict) -> None:
    from hybrid_vpp.training.experiments import (
        ExperimentSpec,
        build_config,
        evaluate_checkpoint,
        evaluation_days,
    )
    from hybrid_vpp.training.train import train

    spec = ExperimentSpec(**json.loads(row["spec_json"]))
    experiment_id = row["experiment_id"]

    heartbeat_dir = Path("runs/heartbeats")
    heartbeat_dir.mkdir(parents=True, exist_ok=True)
    heartbeat_path = heartbeat_dir / f"{experiment_id}.json"
    registry.update(experiment_id, heartbeat_path=str(heartbeat_path))

    # self-sufficient prior-data jobs: build the dataset if it is missing
    # (long phase — heartbeat via thread so the watchdog sees liveness)
    prefill = spec.overrides.get("training.replay_prefill_path") or spec.overrides.get(
        "training.bc_pretrain_path"
    )
    if prefill and not Path(prefill).exists():
        from hybrid_vpp.training.datasets import build_prior_dataset
        from hybrid_vpp.training.research_state import HeartbeatThread

        registry.log_event(experiment_id, "building_prior_dataset", str(prefill))
        with HeartbeatThread(heartbeat_path, experiment_id, "building_dataset"):
            build_prior_dataset(Path("configs/default.yaml"), Path(prefill).parent)

    cfg, config_path = build_config(spec, Path("configs/default.yaml"))

    registry.update(
        experiment_id,
        state="RUNNING",
        heartbeat_path=str(heartbeat_path),
        git_commit=os.popen("git rev-parse --short HEAD").read().strip() or None,
    )
    Heartbeat(experiment_id, utcnow(), os.getpid(), 0, "starting").write(heartbeat_path)

    run_dir = train(config_path, heartbeat_path=heartbeat_path, experiment_id=experiment_id)
    registry.update(experiment_id, state="EVALUATING", run_dir=str(run_dir))
    Heartbeat(experiment_id, utcnow(), os.getpid(), -1, "evaluating").write(heartbeat_path)

    best = run_dir / "best" / "best_model.zip"
    model_path = best if best.exists() else run_dir / "final_model.zip"
    days = evaluation_days(config_path)
    result = evaluate_checkpoint(config_path, model_path, days)

    registry.update(experiment_id, state="VALIDATING")
    problems = validate_training_result(result, config_path, model_path, len(days))
    if problems:
        registry.update(
            experiment_id,
            state="INVALID_RESULT",
            failure_class="CHECKPOINT_CORRUPTION",
            failure_detail="; ".join(problems),
            finished_at=utcnow(),
        )
        registry.log_event(experiment_id, "invalid_result", "; ".join(problems))
        return

    _write_done(
        run_dir, {"experiment_id": experiment_id, "validated_at": utcnow(), "result": result}
    )
    registry.update(
        experiment_id,
        state="COMPLETED",
        result_json=json.dumps(result),
        finished_at=utcnow(),
    )
    registry.log_event(
        experiment_id,
        "completed",
        f"mean {result['mean_revenue_eur']:.0f} median {result['median_revenue_eur']:.0f}",
    )
    # keep the JSONL registry in sync for provenance tooling
    from hybrid_vpp.training.experiments import REGISTRY as JSONL_REGISTRY

    record = {
        "experiment_id": experiment_id,
        "phase": row["phase"],
        "timestamp": utcnow(),
        "algorithm": spec.algorithm,
        "action_mode": spec.action_mode,
        "seed": spec.seed,
        "total_timesteps": spec.total_timesteps,
        "run_dir": str(run_dir),
        "model_path": str(model_path),
        "config_path": str(config_path),
        "validation": result,
        "notes": spec.notes,
        "supervised": True,
    }
    with JSONL_REGISTRY.open("a") as fh:
        fh.write(json.dumps(record) + "\n")


def run_l1_anchor_job(registry: ResearchRegistry, row: dict) -> None:
    """Exact Level-1 anchor: DAA-only + battery, perfect forecasts, MILP."""
    from hybrid_vpp.controllers.base import run_episode
    from hybrid_vpp.controllers.optimization import OptimizationController
    from hybrid_vpp.data.site_profiles import load_site_profiles
    from hybrid_vpp.data.sqlite_market_data import MarketDataStore
    from hybrid_vpp.evaluation.metrics import episode_metrics
    from hybrid_vpp.forecasts.price import PerfectPriceForecast
    from hybrid_vpp.forecasts.renewable import PerfectForesightForecast
    from hybrid_vpp.sim.simulator import Simulator
    from hybrid_vpp.training.experiments import (
        LEVEL1_OVERRIDES,
        ExperimentSpec,
        build_config,
        evaluation_days,
    )

    experiment_id = row["experiment_id"]
    registry.update(experiment_id, state="RUNNING")
    spec = ExperimentSpec("l1-anchor-build", "analysis", overrides=dict(LEVEL1_OVERRIDES))
    cfg, config_path = build_config(spec, Path("configs/default.yaml"))
    store = MarketDataStore(cfg.data, cfg.markets, cfg.synthetic_market)
    profiles = load_site_profiles(cfg.data, cfg.site, store)
    sim = Simulator(cfg, store, profiles)
    controller = OptimizationController(
        cfg,
        PerfectForesightForecast(profiles),
        PerfectPriceForecast(
            {
                "daa": store.daa_prices()["price_eur_per_mwh"],
                "ida1": store.ida_prices("ida1"),
                "ida2": store.ida_prices("ida2"),
                "ida3": store.ida_prices("ida3"),
                "idc": store.idc_indices()["IDFULL"],
            }
        ),
        idc_corrections=False,
    )
    days = evaluation_days(config_path)
    revenues = []
    for day in days:
        run_episode(sim, controller, day)
        revenues.append(episode_metrics(sim)["total_net_revenue_eur"])
    revenue = np.asarray(revenues)
    result = {
        "days": len(days),
        "mean_revenue_eur": float(revenue.mean()),
        "median_revenue_eur": float(np.median(revenue)),
        "total_revenue_eur": float(revenue.sum()),
        "per_day_revenue_eur": [float(v) for v in revenue],
    }
    registry.update(
        experiment_id,
        state="COMPLETED",
        result_json=json.dumps(result),
        finished_at=utcnow(),
    )
    registry.log_event(
        experiment_id, "completed", f"L1 anchor mean {result['mean_revenue_eur']:.0f}"
    )


KIND_HANDLERS = {
    "training": run_training_job,
    "milp_anchor_l1": run_l1_anchor_job,
}


def work_one(registry_path: Path = REGISTRY_PATH) -> str | None:
    """Claim and execute a single job; returns the experiment id (or None)."""
    registry = ResearchRegistry(registry_path)
    row = registry.claim_next(os.getpid())
    if row is None:
        log.info("no queued work")
        return None
    experiment_id = row["experiment_id"]
    log.info("claimed %s (kind=%s)", experiment_id, row["kind"])
    handler = KIND_HANDLERS[row["kind"]]
    try:
        try:
            handler(registry, row)
        except Exception as exc:  # noqa: BLE001
            if "wandb" in type(exc).__module__.lower() or "wandb" in str(exc).lower():
                registry.log_event(experiment_id, "wandb_failure_retry_offline", str(exc))
                os.environ["WANDB_MODE"] = "offline"
                handler(registry, row)
            else:
                raise
    except Exception as exc:  # noqa: BLE001
        detail = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-2000:]}"
        failure_class = classify_exception(exc)
        registry.update(
            row["experiment_id"],
            state="FAILED_RETRYABLE",
            failure_class=failure_class,
            failure_detail=detail,
            finished_at=utcnow(),
        )
        registry.log_event(row["experiment_id"], "failed", f"{failure_class}: {exc}")
    return experiment_id


def classify_exception(exc: Exception) -> str:
    text = f"{type(exc).__name__} {exc}".lower()
    if "memory" in text:
        return "OUT_OF_MEMORY"
    if "nan" in text or "inf" in text:
        return "NAN_OR_INF"
    if "wandb" in text:
        return "WANDB_FAILURE"
    if "sqlite" in text or "database" in text:
        return "DATABASE_FAILURE"
    if "validation" in text or "literal_error" in text or "pydantic" in text:
        return "INVALID_CONFIGURATION"
    if "ledger" in text or "accounting" in text or "retroactive" in text:
        return "ACCOUNTING_INVARIANT_FAILURE"
    return "ENVIRONMENT_EXCEPTION"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    work_one()
