"""RL training entry point (SB3 PPO/SAC) with W&B / TensorBoard tracking.

Edit the CONFIG block and run::

    uv run python -m hybrid_vpp.training.train

Reproducibility: explicit seeds, a config snapshot next to the checkpoints,
periodic deterministic evaluation on the *validation* split (never test),
and best-model selection by validation reward. Resume by pointing
``RESUME_FROM`` at a checkpoint zip.
"""

from __future__ import annotations

import json
import logging
import platform
from datetime import datetime
from pathlib import Path

import yaml

from hybrid_vpp.config.models import ExperimentConfig, load_config

log = logging.getLogger(__name__)


def make_env_fn(cfg: ExperimentConfig, split: str, rank: int, sequential: bool = False):
    def _init():
        from hybrid_vpp.envs.hybrid_vpp_env import HybridVppEnv

        env = HybridVppEnv(cfg, split=split, sequential_days=sequential)
        env.reset(seed=cfg.training.seed + 1000 * rank)
        return env

    return _init


class EpisodeMetricsCallback:
    """Aggregates env episode metrics into the SB3 logger (-> W&B/TB)."""

    def __new__(cls, *args, **kwargs):  # defer BaseCallback import
        from stable_baselines3.common.callbacks import BaseCallback

        class _Callback(BaseCallback):
            KEYS = (
                "total_net_revenue_eur",
                "imbalance_cash_eur",
                "abs_deviation_mwh",
                "wind_curtailed_mwh",
                "pv_curtailed_mwh",
                "equivalent_full_cycles",
                "congestion_wind_curtailed_mwh",
                "congestion_pv_curtailed_mwh",
                "congestion_charge_mwh",
                "corrected_dispatch_intervals",
                "idc_volume_mwh",
                "final_soc",
            )

            def _on_step(self) -> bool:
                for info in self.locals.get("infos", ()):
                    metrics = info.get("episode_metrics")
                    if metrics:
                        for key in self.KEYS:
                            self.logger.record_mean(f"episode/{key}", metrics[key])
                return True

        return _Callback()


class HeartbeatCallback:
    """Writes a machine-readable heartbeat file every ``interval_s`` seconds."""

    def __new__(cls, path: Path, experiment_id: str, interval_s: float = 30.0):
        import os
        import time as _time

        from stable_baselines3.common.callbacks import BaseCallback

        from hybrid_vpp.training.research_state import Heartbeat, utcnow

        class _Callback(BaseCallback):
            def __init__(self) -> None:
                super().__init__()
                self._last = 0.0

            def _on_step(self) -> bool:
                now = _time.time()
                if now - self._last >= interval_s:
                    self._last = now
                    Heartbeat(
                        experiment_id=experiment_id,
                        timestamp=utcnow(),
                        pid=os.getpid(),
                        environment_steps=int(self.num_timesteps),
                        phase="training",
                    ).write(path)
                return True

        return _Callback()


def train(
    config_path: str | Path,
    resume_from: str | None = None,
    heartbeat_path: Path | None = None,
    experiment_id: str | None = None,
) -> Path:
    import torch
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv

    from hybrid_vpp.data.resolver import resolve_market_database

    cfg = load_config(config_path)
    tc = cfg.training
    torch.set_num_threads(2)

    # resolve once up-front so run metadata always names the active data source
    source = resolve_market_database(cfg.data, cfg.markets, cfg.synthetic_market)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(tc.checkpoint_dir) / f"{tc.run_name}_{stamp}_seed{tc.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(json.loads(cfg.model_dump_json())))
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "created": datetime.now().astimezone().isoformat(),
                "host": platform.node(),
                "seed": tc.seed,
                "algorithm": tc.algorithm,
                "resume_from": resume_from,
                "market_data": {
                    "path": str(source.path),
                    "provenance": source.provenance,
                    "problems": list(source.problems),
                },
            },
            indent=2,
        )
    )

    wandb_run = None
    if tc.tracker == "wandb":
        try:
            import wandb

            wandb_run = wandb.init(
                project=tc.wandb_project,
                name=run_dir.name,
                config={
                    **json.loads(cfg.model_dump_json()),
                    "market_data_provenance": source.provenance,
                    "market_data_path": str(source.path),
                },
                sync_tensorboard=True,
                dir=str(run_dir),
            )
        except Exception:
            log.exception("wandb init failed — continuing with tensorboard only")

    # spawn (not fork): forking a parent that already carries torch/wandb
    # threads deadlocks the workers on inherited locks — observed as lanes
    # freezing on their second train() call. Spawned workers re-import and
    # reload the parquet caches (~5 s each), which is an acceptable cost.
    train_env = SubprocVecEnv(
        [make_env_fn(cfg, "train", rank) for rank in range(tc.n_envs)],
        start_method="spawn",
    )
    eval_env = SubprocVecEnv([make_env_fn(cfg, "val", 900, sequential=True)], start_method="spawn")

    from hybrid_vpp.training.algorithms import algo_class, default_kwargs, policy_name

    algo_cls = algo_class(tc.algorithm)
    algo_kwargs = {
        "policy": policy_name(tc.algorithm),
        "env": train_env,
        "seed": tc.seed,
        "verbose": 1,
        "tensorboard_log": str(tc.tensorboard_dir),
    }
    algo_kwargs.update(default_kwargs(tc.algorithm))
    algo_kwargs.update(tc.algo_kwargs)
    if tc.policy_kwargs:
        algo_kwargs["policy_kwargs"] = {**algo_kwargs.get("policy_kwargs", {}), **tc.policy_kwargs}

    if resume_from:
        model = algo_cls.load(resume_from, env=train_env)
        log.info("resumed from %s", resume_from)
    else:
        model = algo_cls(**algo_kwargs)

    eval_freq = max(tc.eval_freq // tc.n_envs, 1)
    callbacks = [
        EvalCallback(
            eval_env,
            best_model_save_path=str(run_dir / "best"),
            log_path=str(run_dir / "eval"),
            eval_freq=eval_freq,
            n_eval_episodes=tc.n_eval_episodes,
            deterministic=True,
        ),
        CheckpointCallback(
            save_freq=eval_freq, save_path=str(run_dir / "checkpoints"), name_prefix="model"
        ),
        EpisodeMetricsCallback(),
    ]
    if heartbeat_path is not None:
        callbacks.append(HeartbeatCallback(Path(heartbeat_path), experiment_id or run_dir.name))

    model.learn(
        total_timesteps=tc.total_timesteps,
        callback=callbacks,
        tb_log_name=run_dir.name,
        progress_bar=False,
    )
    model.save(run_dir / "final_model")
    train_env.close()
    eval_env.close()
    if wandb_run is not None:
        wandb_run.finish()
    log.info("training complete: %s", run_dir)
    return run_dir


# --------------------------------------------------------------------------
# CONFIG — edit and run as a module (no CLI flags by design)
# --------------------------------------------------------------------------
CONFIG_PATH = Path("configs/default.yaml")
RESUME_FROM: str | None = None

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    train(CONFIG_PATH, RESUME_FROM)
