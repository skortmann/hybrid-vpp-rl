"""Hierarchical bootstrap over training seeds and contiguous day blocks.

Seed-day observations are not independent: days share market regimes
(serial correlation) and seeds share training data. The bootstrap here
resamples both levels — training seeds with replacement, then contiguous
day blocks within each resampled seed (moving-block bootstrap) — so the
reported uncertainty reflects seed variation *and* day-to-day market
variation without treating every seed-day pair as independent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _moving_block_indices(n_days: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    """Concatenated random contiguous blocks, trimmed to ``n_days`` indices."""
    if not 1 <= block_len <= n_days:
        raise ValueError(f"block_len must be in [1, {n_days}], got {block_len}")
    n_blocks = int(np.ceil(n_days / block_len))
    starts = rng.integers(0, n_days - block_len + 1, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block_len) for s in starts])
    return idx[:n_days]


def hierarchical_bootstrap(
    per_day: pd.DataFrame,
    reference: pd.Series,
    n_boot: int = 10_000,
    block_len: int = 7,
    seed: int = 0,
) -> dict[str, float]:
    """Paired candidate-minus-reference statistics with two-level resampling.

    ``per_day``: days x seeds revenue table for one candidate formulation
    (each column is an independently trained seed). ``reference``: per-day
    revenue of the comparison controller on the same days.
    """
    ref = reference.reindex(per_day.index)
    if ref.isna().any():
        raise ValueError("reference series does not cover all candidate days")
    diff = per_day.sub(ref, axis=0).to_numpy()  # (n_days, n_seeds)
    n_days, n_seeds = diff.shape
    rng = np.random.default_rng(seed)

    stats = np.empty(n_boot)
    for i in range(n_boot):
        seed_idx = rng.integers(0, n_seeds, size=n_seeds)
        seed_means = np.empty(n_seeds)
        for j, s in enumerate(seed_idx):
            day_idx = _moving_block_indices(n_days, block_len, rng)
            seed_means[j] = diff[day_idx, s].mean()
        stats[i] = seed_means.mean()

    per_seed_means = diff.mean(axis=0)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    trimmed = np.sort(diff.mean(axis=1))[n_days // 4 : n_days - n_days // 4]
    return {
        "mean_diff": float(diff.mean()),
        "median_diff": float(np.median(diff)),
        "iq_mean_diff": float(trimmed.mean()),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "p_outperform": float((stats > 0).mean()),
        "day_win_rate": float((diff > 0).mean()),
        "seed_std": float(per_seed_means.std(ddof=1)) if n_seeds > 1 else 0.0,
        "day_std": float(diff.mean(axis=1).std(ddof=1)),
        "n_days": n_days,
        "n_seeds": n_seeds,
        "block_len": block_len,
        "n_boot": n_boot,
    }
