"""Safety-gate rollouts: RL ensemble proposals gated against rule-based.

Evaluates the safety gates of :mod:`hybrid_vpp.envs.safety_gate` around
the ensemble proposal on every validation day. Disagreement thresholds
(Gates A and B) are calibrated per held-out block from the *other*
blocks' disagreement distribution (cached by
:mod:`hybrid_vpp.evaluation.ensemble_eval`) — the held-out block never
influences its own thresholds. Gate C (bounded residual) needs no
calibration and uses fixed per-run bounds.

Run as ``uv run python -m hybrid_vpp.evaluation.gate_eval`` after
``ensemble_eval``; results are cached next to the other candidates.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------- CONFIG

SPLIT = "val"
N_BLOCKS = 6
CONFIG_PATH = Path("runs/experiments/V6-hybrid-sac-strategic-seed0.yaml")
PROPOSAL_MODE = "mean"  # ensemble variant that proposes actions
QUANTILES = (0.5, 0.8)  # threshold calibration quantiles (Gates A and B)
RESIDUAL_BOUNDS = (0.1, 0.2, 0.4)  # per-dim residual bounds (Gate C)
CACHE_DIR = Path("artifacts/robust_selection/cache")
N_PROCESSES = 5

# ------------------------------------------------------------------------


def gate_variants() -> list[dict]:
    variants = []
    for q in QUANTILES:
        variants.append(
            {"name": f"gate_a_q{int(q * 100)}", "mode": "disagreement_threshold", "quantile": q}
        )
        variants.append(
            {"name": f"gate_b_q{int(q * 100)}", "mode": "confidence_scaling", "quantile": q}
        )
    for r in RESIDUAL_BOUNDS:
        variants.append({"name": f"gate_c_r{r:g}", "mode": "bounded_residual", "bound": r})
    return variants


def _block_thresholds(split: str, n_blocks: int, quantile: float) -> dict[int, dict[str, float]]:
    """LOBO thresholds: block index -> event type -> u threshold."""
    from hybrid_vpp.envs.safety_gate import disagreement_thresholds
    from hybrid_vpp.evaluation.blocked_validation import contiguous_blocks

    cache = CACHE_DIR / split / f"ensemble_{PROPOSAL_MODE}.json"
    data = json.loads(cache.read_text())
    dis_by_day = {day: rec["disagreement"] for day, rec in data.items()}
    days_index = pd.DatetimeIndex(sorted(pd.Timestamp(d) for d in dis_by_day))
    blocks = contiguous_blocks(days_index, n_blocks)
    out = {}
    for b, block in enumerate(blocks):
        inner = [str(d.date()) for d in days_index if d not in set(block)]
        out[b] = disagreement_thresholds(dis_by_day, inner, quantile)
    return out


def run_gate(variant: dict, split: str = SPLIT, n_blocks: int = N_BLOCKS) -> dict[str, dict]:
    """Roll one gate variant over all days of ``split``."""
    import torch

    from hybrid_vpp.config.models import load_config
    from hybrid_vpp.envs.ensemble import PolicyEnsemble, disagreement
    from hybrid_vpp.envs.hybrid_vpp_env import HybridVppEnv
    from hybrid_vpp.envs.safety_gate import SafetyGate, rule_equivalent_action
    from hybrid_vpp.envs.strategic import STRATEGIC_MASKS
    from hybrid_vpp.evaluation.blocked_validation import contiguous_blocks
    from hybrid_vpp.evaluation.checkpoint_matrix import METRIC_KEYS
    from hybrid_vpp.evaluation.ensemble_eval import member_paths
    from hybrid_vpp.training.algorithms import algo_class

    torch.set_num_threads(1)
    cfg = load_config(CONFIG_PATH)
    env = HybridVppEnv(cfg, split=split)
    models = [algo_class("sac").load(p) for p in member_paths()]
    ensemble = PolicyEnsemble(models, mode=PROPOSAL_MODE)
    rule_action = rule_equivalent_action(cfg.episode.strategic_gain_max)
    active_dims = {ev.name: dims for ev, dims in STRATEGIC_MASKS.items()}
    blocks = contiguous_blocks(pd.DatetimeIndex(env.valid_days), n_blocks)
    block_of = {day: b for b, block in enumerate(blocks) for day in block}

    thresholds = (
        _block_thresholds(split, n_blocks, variant["quantile"])
        if variant["mode"] in ("disagreement_threshold", "confidence_scaling")
        else None
    )

    out: dict[str, dict] = {}
    for day in env.valid_days:
        if thresholds is not None:
            gate = SafetyGate(variant["mode"], rule_action, u_thresholds=thresholds[block_of[day]])
        else:
            gate = SafetyGate(variant["mode"], rule_action, max_residual=variant["bound"])
        obs, info = env.reset(options={"day": day})
        records, done = [], False
        while not done:
            proposal, members = ensemble.predict(obs)
            ev = info["event_type"]
            u = disagreement(members, active_dims.get(ev))["u"]
            action, record = gate.apply(proposal, ev, u=u)
            if ev != "PHYSICAL_DISPATCH":  # dispatch is deterministic rule-based
                records.append(record)
            obs, _, done, _, info = env.step(action)
        metrics = info["episode_metrics"]
        out[str(day.date())] = {
            "metrics": {k: metrics[k] for k in METRIC_KEYS if k in metrics},
            "fallback_rate": float(np.mean([r["fallback"] for r in records])),
            "alpha_mean": float(np.mean([r["alpha"] for r in records])),
        }
    return out


def _worker(variant: dict) -> str:
    import logging

    logging.basicConfig(level=logging.WARNING)
    result = run_gate(variant)
    cache = CACHE_DIR / SPLIT / f"{variant['name']}.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp")
    tmp.write_text(json.dumps(result))
    tmp.replace(cache)
    return variant["name"]


def main() -> None:
    import multiprocessing as mp

    todo = [v for v in gate_variants() if not (CACHE_DIR / SPLIT / f"{v['name']}.json").exists()]
    if not todo:
        print("all gate variants cached")
        return
    print(f"evaluating {len(todo)} gate variants with {N_PROCESSES} processes")
    with mp.get_context("spawn").Pool(min(N_PROCESSES, len(todo))) as pool:
        for name in pool.imap_unordered(_worker, todo):
            data = json.loads((CACHE_DIR / SPLIT / f"{name}.json").read_text())
            revenues = [r["metrics"]["total_net_revenue_eur"] for r in data.values()]
            fallback = float(np.mean([r["fallback_rate"] for r in data.values()]))
            print(
                f"  {name}: mean {np.mean(revenues):,.0f} median {np.median(revenues):,.0f} "
                f"fallback {fallback:.1%}",
                flush=True,
            )


if __name__ == "__main__":
    main()
