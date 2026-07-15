"""Policy and formulation diagnostics: action use, rewards, projections.

Rolls a trained policy (or a zero-action baseline) over fixed validation
days and decomposes, per market-event type:

* step counts and reward (cash) contributions,
* action statistics on the *active* dimensions (mean |a|, saturation),
* feasibility-projection frequency and magnitude at dispatch events,
* ledger components (per market, imbalance, costs).

Used to answer: which parts of the action vector matter, where does the
money move, and how often does the feasibility layer overwrite the policy.

Edit the CONFIG block and run::

    uv run python -m hybrid_vpp.evaluation.diagnostics
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from hybrid_vpp.config.models import load_config

log = logging.getLogger(__name__)

SATURATION_THRESHOLD = 0.95


def diagnose(
    config_path: Path,
    model_path: Path | None,
    n_days: int = 10,
    split: str = "val",
) -> dict:
    """Run the diagnosis; ``model_path=None`` uses a zero action everywhere."""
    from hybrid_vpp.envs.hybrid_vpp_env import HybridVppEnv

    cfg = load_config(config_path)
    model = None
    if model_path is not None:
        from hybrid_vpp.training.algorithms import algo_class

        model = algo_class(cfg.training.algorithm).load(model_path)

    env = HybridVppEnv(cfg, split=split)
    days = env.valid_days[:n_days]

    stats: dict[str, dict] = defaultdict(
        lambda: {
            "steps": 0,
            "reward_keur": 0.0,
            "abs_action_sum": 0.0,
            "active_entries": 0,
            "saturated_entries": 0,
        }
    )
    dispatch = {
        "requested_vs_applied_bess_mw": [],
        "corrected_steps": 0,
        "congestion_charge_mwh": 0.0,
        "congestion_curtail_mwh": 0.0,
        "total_steps": 0,
    }
    ledger_totals: dict[str, float] = defaultdict(float)

    for day in days:
        obs, info = env.reset(options={"day": day})
        done = False
        while not done:
            event_type = info["event_type"]
            mask = info["action_mask"].astype(bool)
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = np.zeros(env.action_space.shape, np.float32)
            obs, reward, done, _, info = env.step(action)

            s = stats[event_type]
            s["steps"] += 1
            s["reward_keur"] += float(reward)
            active = action[mask]
            s["active_entries"] += int(mask.sum())
            s["abs_action_sum"] += float(np.abs(active).sum())
            s["saturated_entries"] += int((np.abs(active) > SATURATION_THRESHOLD).sum())

        # per-episode physical audit
        for record in env.sim.dispatch_records.values():
            d = record.dispatch
            dispatch["total_steps"] += 1
            if d.was_corrected:
                dispatch["corrected_steps"] += 1
            dispatch["requested_vs_applied_bess_mw"].append(
                abs(d.requested_bess_power_mw - d.bess_power_mw)
            )
            dispatch["congestion_charge_mwh"] += d.congestion_charge_mw * 0.25
            dispatch["congestion_curtail_mwh"] += (
                d.congestion_wind_curtail_mw + d.congestion_pv_curtail_mw
            ) * 0.25
        for component, amount in env.sim.ledger.by_component().items():
            ledger_totals[component] += amount

    report = {
        "config": str(config_path),
        "model": str(model_path) if model_path else "zero-action",
        "action_mode": cfg.episode.action_mode,
        "days": len(days),
        "per_event": {
            event: {
                "steps": s["steps"],
                "reward_keur_total": round(s["reward_keur"], 2),
                "reward_share": 0.0,  # filled below
                "mean_abs_active_action": round(
                    s["abs_action_sum"] / max(s["active_entries"], 1), 4
                ),
                "saturation_rate": round(s["saturated_entries"] / max(s["active_entries"], 1), 4),
            }
            for event, s in sorted(stats.items())
        },
        "dispatch_projection": {
            "corrected_fraction": round(
                dispatch["corrected_steps"] / max(dispatch["total_steps"], 1), 4
            ),
            "mean_bess_projection_mw": round(
                float(np.mean(dispatch["requested_vs_applied_bess_mw"] or [0.0])), 3
            ),
            "congestion_charge_mwh": round(dispatch["congestion_charge_mwh"], 2),
            "congestion_curtail_mwh": round(dispatch["congestion_curtail_mwh"], 2),
        },
        "ledger_totals_eur": {k: round(v, 0) for k, v in sorted(ledger_totals.items()) if v},
    }
    total_reward = sum(abs(e["reward_keur_total"]) for e in report["per_event"].values()) or 1.0
    for e in report["per_event"].values():
        e["reward_share"] = round(abs(e["reward_keur_total"]) / total_reward, 3)
    return report


# --------------------------------------------------------------------------
# CONFIG — edit and run as a module (no CLI flags by design)
# --------------------------------------------------------------------------
CONFIG_PATH = Path("configs/default.yaml")
MODEL_PATH: Path | None = None  # None -> zero-action baseline
N_DAYS = 10
OUTPUT = Path("experiments/diagnostics.json")

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    result = diagnose(CONFIG_PATH, MODEL_PATH, N_DAYS)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
