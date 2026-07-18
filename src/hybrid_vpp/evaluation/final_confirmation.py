"""Phase 8: one-shot reused-test confirmation of the promoted controller.

Runs the locked candidate — mean-action ensemble of the five frozen
eval-best checkpoints, bounded residual 0.1, deterministic rule-based
dispatch — once over the 98-day test split, and compares it against the
*recorded* baselines of the previous phase (`artifacts/test_evaluation.json`)
with a moving-block bootstrap. The test split was already examined by
the previous research phase; this is a reused-test confirmation, not an
untouched-test claim. The protocol was pre-registered in
the research log (repository tag ``robust-rl-final``) before any test contact.

Run as ``uv run python -m hybrid_vpp.evaluation.final_confirmation``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------- CONFIG

CONFIG_PATH = Path("runs/experiments/V6-hybrid-sac-strategic-seed0.yaml")
RESIDUAL_BOUND = 0.1
BLOCK_LEN = 7
N_BOOT = 10_000
RECORDED_TEST = Path("artifacts/test_evaluation.json")
OUT_PATH = Path("artifacts/robust_selection/final_test_confirmation.json")

# ------------------------------------------------------------------------


def run_locked_candidate() -> pd.Series:
    """Deterministic pass of the locked construction over the test split."""
    import torch

    from hybrid_vpp.config.models import load_config
    from hybrid_vpp.envs.ensemble import PolicyEnsemble
    from hybrid_vpp.envs.hybrid_vpp_env import HybridVppEnv
    from hybrid_vpp.envs.safety_gate import SafetyGate, rule_equivalent_action
    from hybrid_vpp.evaluation.ensemble_eval import member_paths
    from hybrid_vpp.training.algorithms import algo_class

    torch.set_num_threads(1)
    cfg = load_config(CONFIG_PATH)
    env = HybridVppEnv(cfg, split="test")
    models = [algo_class("sac").load(p) for p in member_paths()]
    ensemble = PolicyEnsemble(models, mode="mean")
    gate = SafetyGate(
        "bounded_residual",
        rule_equivalent_action(cfg.episode.strategic_gain_max),
        max_residual=RESIDUAL_BOUND,
    )
    revenues = {}
    for day in env.valid_days:
        obs, info = env.reset(options={"day": day})
        done = False
        while not done:
            proposal, _ = ensemble.predict(obs)
            action, _rec = gate.apply(proposal, info["event_type"])
            obs, _, done, _, info = env.step(action)
        revenues[pd.Timestamp(day)] = info["episode_metrics"]["total_net_revenue_eur"]
        print(f"  {day.date()}: {revenues[pd.Timestamp(day)]:,.0f}", flush=True)
    return pd.Series(revenues).sort_index()


def main() -> None:
    from hybrid_vpp.evaluation.blocked_validation import cvar
    from hybrid_vpp.evaluation.hierarchical_bootstrap import hierarchical_bootstrap

    if OUT_PATH.exists():
        raise SystemExit(f"{OUT_PATH} already exists — the one-shot confirmation is locked")

    recorded = json.loads(RECORDED_TEST.read_text())
    days = pd.DatetimeIndex(pd.Timestamp(d) for d in recorded["days"])
    baselines = {k: pd.Series(v, index=days) for k, v in recorded["baselines"].items()}

    candidate = run_locked_candidate()
    if not candidate.index.equals(days):
        raise RuntimeError("test day mismatch vs recorded baselines")

    regret = candidate - baselines["rule_based"]
    boot = hierarchical_bootstrap(
        candidate.to_frame("locked"), baselines["rule_based"], N_BOOT, BLOCK_LEN, seed=0
    )
    milp = baselines["milp_info"]
    result = {
        "candidate": "gate_c_r0.1 over five frozen eval-best checkpoints",
        "protocol": "pre-registered one-shot reused-test confirmation",
        "n_days": int(len(candidate)),
        "mean": float(candidate.mean()),
        "median": float(candidate.median()),
        "baselines_mean": {k: float(v.mean()) for k, v in baselines.items()},
        "baselines_median": {k: float(v.median()) for k, v in baselines.items()},
        "paired_vs_rule_based": {
            "mean": boot["mean_diff"],
            "median": float(regret.median()),
            "ci95": [boot["ci95_low"], boot["ci95_high"]],
            "p_outperform": boot["p_outperform"],
            "day_win_rate": boot["day_win_rate"],
            "cvar10_regret": cvar(regret, 0.10),
            "downside_exposure": float((-regret).clip(lower=0).mean()),
            "max_daily_loss": float((-regret).max()),
        },
        "median_info_gap_pct": float((((milp - candidate) / milp.abs()).median()) * 100),
        "per_day": {str(d.date()): float(v) for d, v in candidate.items()},
    }
    OUT_PATH.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "per_day"}, indent=1))


if __name__ == "__main__":
    main()
