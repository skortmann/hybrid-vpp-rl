"""Ensemble rollouts over blocked validation folds, with disagreement logs.

Evaluates action-space ensembles of the five trained SAC policies
(:class:`hybrid_vpp.envs.ensemble.PolicyEnsemble`) on every day of the
chosen split, logging per-event policy disagreement for the safety-gate
analysis. The validation-weighted variant derives its weights per
held-out block from the *other* blocks only (leave-one-block-out), using
the cached checkpoint matrix — final-test information is never used.

Run as ``uv run python -m hybrid_vpp.evaluation.ensemble_eval`` after
``hybrid_vpp.evaluation.checkpoint_matrix``; results are cached next to
the checkpoint caches so the analysis sees one consistent table.
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
MEMBER_LABEL = "best"  # which checkpoint of each seed joins the ensemble
MODES = ("mean", "median", "trimmed_mean", "weighted")
SOFTMAX_TEMPERATURE = 1.0  # in units of the member-mean std (weighted mode)
CACHE_DIR = Path("artifacts/robust_selection/cache")
OUT_DIR = Path("artifacts/robust_selection")

# ------------------------------------------------------------------------


def member_paths(label: str = MEMBER_LABEL) -> list[str]:
    """Checkpoint paths of the ensemble members (one per seed)."""
    from hybrid_vpp.evaluation.checkpoint_matrix import list_checkpoints

    refs = [r for r in list_checkpoints() if r["label"] == label]
    if len(refs) < 2:
        raise RuntimeError(f"need at least two members with label {label!r}")
    return [r["path"] for r in refs]


def lobo_weights(
    split: str = SPLIT,
    label: str = MEMBER_LABEL,
    n_blocks: int = N_BLOCKS,
    temperature: float = SOFTMAX_TEMPERATURE,
) -> dict[int, np.ndarray]:
    """Per-block simplex weights from the other blocks' member means.

    Weight of member i for block b: softmax of the z-scored mean revenue
    of member i over all validation days *outside* block b.
    """
    from hybrid_vpp.evaluation.blocked_validation import contiguous_blocks
    from hybrid_vpp.evaluation.checkpoint_matrix import load_per_day_table

    table = load_per_day_table(split)
    members = [c for c in table.columns if c.startswith("ckpt_") and c.endswith(f"_{label}")]
    if len(members) < 2:
        raise RuntimeError(f"checkpoint matrix has no members with label {label!r}")
    blocks = contiguous_blocks(table.index, n_blocks)
    weights = {}
    for b, block in enumerate(blocks):
        inner = table.loc[~table.index.isin(block), members]
        means = inner.mean().to_numpy()
        std = means.std(ddof=0)
        z = (means - means.mean()) / std if std > 0 else np.zeros_like(means)
        e = np.exp(z / temperature)
        weights[b] = e / e.sum()
    return weights


def run_ensemble(
    mode: str,
    split: str = SPLIT,
    weights_per_block: dict[int, np.ndarray] | None = None,
    n_blocks: int = N_BLOCKS,
) -> dict[str, dict]:
    """Roll one ensemble variant over all days of ``split``."""
    import torch

    from hybrid_vpp.config.models import load_config
    from hybrid_vpp.envs.ensemble import PolicyEnsemble, disagreement
    from hybrid_vpp.envs.hybrid_vpp_env import HybridVppEnv
    from hybrid_vpp.envs.strategic import STRATEGIC_MASKS
    from hybrid_vpp.evaluation.blocked_validation import contiguous_blocks
    from hybrid_vpp.evaluation.checkpoint_matrix import METRIC_KEYS
    from hybrid_vpp.training.algorithms import algo_class

    torch.set_num_threads(1)
    cfg = load_config(CONFIG_PATH)
    env = HybridVppEnv(cfg, split=split)
    models = [algo_class("sac").load(p) for p in member_paths()]
    active_dims = {ev.name: dims for ev, dims in STRATEGIC_MASKS.items()}
    blocks = contiguous_blocks(pd.DatetimeIndex(env.valid_days), n_blocks)
    block_of = {day: b for b, block in enumerate(blocks) for day in block}

    out: dict[str, dict] = {}
    for day in env.valid_days:
        if mode == "weighted":
            if weights_per_block is None:
                raise ValueError("weighted mode needs weights_per_block")
            ensemble = PolicyEnsemble(
                models, mode="weighted", weights=weights_per_block[block_of[day]]
            )
        else:
            ensemble = PolicyEnsemble(models, mode=mode)
        obs, info = env.reset(options={"day": day})
        dis: dict[str, list[float]] = {}
        done = False
        while not done:
            action, members = ensemble.predict(obs)
            ev = info["event_type"]
            dis.setdefault(ev, []).append(round(disagreement(members, active_dims.get(ev))["u"], 6))
            obs, _, done, _, info = env.step(action)
        metrics = info["episode_metrics"]
        market_u = [u for ev, us in dis.items() if ev != "PHYSICAL_DISPATCH" for u in us]
        out[str(day.date())] = {
            "metrics": {k: metrics[k] for k in METRIC_KEYS if k in metrics},
            "disagreement": dis,
            "u_market_mean": float(np.mean(market_u)),
            "u_market_max": float(np.max(market_u)),
        }
    return out


def main() -> None:
    weights = lobo_weights()
    print("LOBO weights per block:")
    for b, w in weights.items():
        print(f"  block {b}: {np.round(w, 3).tolist()}")
    for mode in MODES:
        cache = CACHE_DIR / SPLIT / f"ensemble_{mode}.json"
        if cache.exists():
            print(f"cached: {cache.name}")
            continue
        result = run_ensemble(mode, weights_per_block=weights if mode == "weighted" else None)
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".tmp")
        tmp.write_text(json.dumps(result))
        tmp.replace(cache)
        revenues = [r["metrics"]["total_net_revenue_eur"] for r in result.values()]
        print(
            f"ensemble_{mode}: mean {np.mean(revenues):,.0f} median {np.median(revenues):,.0f}",
            flush=True,
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "ensemble_weights.json").write_text(
        json.dumps({str(b): w.tolist() for b, w in weights.items()}, indent=2)
    )


if __name__ == "__main__":
    main()
