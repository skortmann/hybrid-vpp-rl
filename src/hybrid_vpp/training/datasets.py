"""Prior-trajectory dataset builder for offline and offline-to-online RL.

Rolls a family of behavior policies over TRAINING-split days only and
records full transitions in a fixed action schema. In the strategic schema
(act-v5) several controller behaviors are *exactly* expressible as
parameter vectors, so the recorded tensor actions reproduce the behavior
with no approximation:

* ``do_nothing``      — sell the forecast, idle battery, never curtail
* ``rule_based``      — the rule-based controller (mid-range parameters)
* ``conservative``    — 80% coverage, half-strength corrections
* ``high_turnover``   — full corrections plus discharge bias
* ``random_k``        — seeded uniform strategic vectors (diverse quality)

Each transition stores: observation, action, reward (true economic, kEUR),
next observation, termination, event type, action mask, projection
distance at dispatch, controller id, episode day, and step index —
the schema required for IQL/CQL/RLPD-style training. Datasets are written
as ``.npz`` under ``data/generated/prior_trajectories/`` with a JSON
manifest (schema versions, config hash, seed, split bounds).

Edit the CONFIG block and run::

    uv run python -m hybrid_vpp.training.datasets
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

from hybrid_vpp.config.models import load_config

log = logging.getLogger(__name__)

#: named behaviors as constant strategic parameter vectors
#: [daa_coverage, arbitrage, ida_gain, idc_gain, tracking, curtail_thr, soc_bias]
BEHAVIORS: dict[str, np.ndarray] = {
    "do_nothing": np.array([2 / 1.2 - 1, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0]),
    "rule_based": np.array([2 / 1.2 - 1, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]),
    "conservative": np.array([0.8 * 2 / 1.2 - 1, 0.0, 0.0, 0.0, 1.0, -0.2, 0.0]),
    "high_turnover": np.array([2 / 1.2 - 1, 1.0, 1.0, 1.0, 1.0, 0.2, 0.4]),
}


def record_controller(env, action_fn, controller_id: str, days) -> dict[str, np.ndarray]:
    """Roll one behavior over days; return stacked transition arrays."""
    buffers: dict[str, list] = {
        k: []
        for k in (
            "obs",
            "action",
            "reward",
            "next_obs",
            "done",
            "event_type",
            "action_mask",
            "projection_mw",
            "day",
            "step",
        )
    }
    for day in days:
        obs, info = env.reset(options={"day": day})
        done, step = False, 0
        while not done:
            action = action_fn(obs, info)
            next_obs, reward, done, _, next_info = env.step(action)
            record = env.sim.dispatch_records
            projection = 0.0
            if info["event_type"] == "PHYSICAL_DISPATCH" and record:
                d = list(record.values())[-1].dispatch
                projection = (
                    abs(d.requested_bess_power_mw - d.bess_power_mw)
                    + abs(d.requested_wind_curtail_mw - d.wind_curtail_mw)
                    + abs(d.requested_pv_curtail_mw - d.pv_curtail_mw)
                )
            buffers["obs"].append(obs)
            buffers["action"].append(np.asarray(action, dtype=np.float32))
            buffers["reward"].append(np.float32(reward))
            buffers["next_obs"].append(next_obs if not done else np.zeros_like(obs))
            buffers["done"].append(done)
            buffers["event_type"].append(info["event_type"])
            buffers["action_mask"].append(info["action_mask"])
            buffers["projection_mw"].append(np.float32(projection))
            buffers["day"].append(str(day.date()))
            buffers["step"].append(step)
            obs, info = next_obs, next_info
            step += 1
    arrays = {k: np.asarray(v) for k, v in buffers.items()}
    arrays["controller"] = np.full(len(arrays["reward"]), controller_id, dtype=object)
    return arrays


def build_prior_dataset(
    config_path: Path,
    out_dir: Path,
    n_days: int = 120,
    n_random_policies: int = 4,
    seed: int = 0,
) -> Path:
    """Build the training-split prior dataset in the strategic action schema."""
    from hybrid_vpp.envs.actions import ACTION_SCHEMA_VERSIONS
    from hybrid_vpp.envs.hybrid_vpp_env import HybridVppEnv

    cfg = load_config(config_path)
    cfg.episode.action_mode = "strategic"
    env = HybridVppEnv(cfg, split="train")
    rng = np.random.default_rng(seed)
    days = env.valid_days[
        np.sort(
            rng.choice(len(env.valid_days), size=min(n_days, len(env.valid_days)), replace=False)
        )
    ]

    parts = []
    for name, params in BEHAVIORS.items():
        vector = params.astype(np.float32)
        parts.append(record_controller(env, lambda o, i, v=vector: v, name, days))
        log.info(
            "%s: %d transitions, mean daily revenue %.0f EUR",
            name,
            len(parts[-1]["reward"]),
            parts[-1]["reward"].sum() * 1e3 / len(days),
        )
    for k in range(n_random_policies):
        policy_rng = np.random.default_rng(seed + 100 + k)
        base = policy_rng.uniform(-1, 1, size=7).astype(np.float32)

        def jitter(obs, info, b=base, r=policy_rng):
            return np.clip(b + r.normal(0, 0.15, size=7), -1, 1).astype(np.float32)

        parts.append(record_controller(env, jitter, f"random_{k}", days))
        log.info(
            "random_%d: mean daily revenue %.0f EUR", k, parts[-1]["reward"].sum() * 1e3 / len(days)
        )

    merged = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / "prior_strategic_v1.npz"
    np.savez_compressed(
        data_path, **{k: (v.astype("U32") if v.dtype == object else v) for k, v in merged.items()}
    )
    manifest = {
        "created": datetime.now().astimezone().isoformat(),
        "action_schema": ACTION_SCHEMA_VERSIONS["strategic"],
        "observation_size": int(env.observation_space.shape[0]),
        "split": "train",
        "days": int(len(days)),
        "first_day": str(days[0].date()),
        "last_day": str(days[-1].date()),
        "seed": seed,
        "behaviors": list(BEHAVIORS) + [f"random_{k}" for k in range(n_random_policies)],
        "transitions": int(len(merged["reward"])),
    }
    (out_dir / "prior_strategic_v1.json").write_text(json.dumps(manifest, indent=2))
    log.info("wrote %s (%d transitions)", data_path, manifest["transitions"])
    return data_path


# --------------------------------------------------------------------------
# CONFIG — edit and run as a module (no CLI flags by design)
# --------------------------------------------------------------------------
CONFIG_PATH = Path("configs/default.yaml")
OUT_DIR = Path("data/generated/prior_trajectories")
N_DAYS = 120
N_RANDOM_POLICIES = 4
SEED = 0

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    build_prior_dataset(CONFIG_PATH, OUT_DIR, N_DAYS, N_RANDOM_POLICIES, SEED)
