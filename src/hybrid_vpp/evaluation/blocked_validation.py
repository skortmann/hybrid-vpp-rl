"""Blocked temporal validation and predefined policy-selection rules.

The validation period is split into contiguous, chronologically ordered
blocks (no shuffling — serial and seasonal structure is preserved).
Candidate policies are compared through fold-level metrics and a set of
selection rules that are fixed *before* any held-out data is examined.
Selection-rule reliability is measured by leave-one-block-out: a rule
selects on all-but-one block and is scored on the held-out block.

Everything here is pure computation on per-day revenue tables; episode
rollouts live in :mod:`hybrid_vpp.evaluation.checkpoint_matrix`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------- blocks


def contiguous_blocks(days: pd.DatetimeIndex, n_blocks: int) -> list[pd.DatetimeIndex]:
    """Split ``days`` into ``n_blocks`` contiguous, near-equal, ordered blocks."""
    if not days.is_monotonic_increasing:
        raise ValueError("days must be sorted chronologically")
    if days.has_duplicates:
        raise ValueError("days must be unique")
    if not 1 <= n_blocks <= len(days):
        raise ValueError(f"n_blocks must be in [1, {len(days)}], got {n_blocks}")
    return [pd.DatetimeIndex(chunk) for chunk in np.array_split(np.asarray(days), n_blocks)]


# --------------------------------------------------------------------- metrics


def cvar(values: np.ndarray | pd.Series, alpha: float = 0.10) -> float:
    """Expected shortfall: mean of the lowest ``alpha`` tail of ``values``."""
    v = np.sort(np.asarray(values, dtype=float))
    if v.size == 0:
        raise ValueError("cvar of empty array")
    k = max(1, int(np.ceil(alpha * v.size)))
    return float(v[:k].mean())


@dataclass(frozen=True)
class SelectionWeights:
    """Coefficients of the risk-adjusted selection score (EUR/day units)."""

    fold_std: float = 0.5
    cvar_regret: float = 0.5
    negative_regret: float = 0.25
    cvar_alpha: float = 0.10


def candidate_metrics(
    revenue: pd.Series,
    reference: pd.Series,
    blocks: list[pd.DatetimeIndex],
    milp: pd.Series | None = None,
    cvar_alpha: float = 0.10,
) -> dict[str, float]:
    """Fold-level and daily metrics for one candidate.

    ``revenue``/``reference`` are per-day EUR series on the same index;
    ``reference`` is the fallback controller (rule-based). Regret is
    ``revenue - reference`` (positive = candidate ahead).
    """
    revenue, reference = revenue.astype(float), reference.reindex(revenue.index).astype(float)
    if reference.isna().any():
        raise ValueError("reference series does not cover all candidate days")
    fold_means = np.array([revenue.loc[revenue.index.isin(b)].mean() for b in blocks])
    regret = revenue - reference
    out = {
        "mean": float(revenue.mean()),
        "median": float(revenue.median()),
        "daily_std": float(revenue.std(ddof=1)),
        "fold_std": float(fold_means.std(ddof=1)) if len(fold_means) > 1 else 0.0,
        "worst_fold": float(fold_means.min()),
        "cvar_revenue": cvar(revenue, cvar_alpha),
        "mean_regret": float(regret.mean()),
        "median_regret": float(regret.median()),
        "cvar_regret": cvar(regret, cvar_alpha),
        "p_beat_reference": float((regret > 0).mean()),
        "upside_capture": float(regret.clip(lower=0).mean()),
        "downside_exposure": float((-regret).clip(lower=0).mean()),
        "max_daily_loss_vs_reference": float((-regret).max()),
    }
    if milp is not None:
        gap = (milp.reindex(revenue.index) - revenue) / milp.reindex(revenue.index).abs()
        out["mean_regret_vs_milp"] = float((revenue - milp.reindex(revenue.index)).mean())
        out["median_info_gap_pct"] = float(gap.median() * 100)
    return out


def selection_score(metrics: dict[str, float], weights: SelectionWeights) -> float:
    """Risk-adjusted score: mean − λσ·fold_std − λCVaR·tail-loss − λR·mean shortfall."""
    tail_loss = max(0.0, -metrics["cvar_regret"])
    shortfall = max(0.0, -metrics["mean_regret"])
    return (
        metrics["mean"]
        - weights.fold_std * metrics["fold_std"]
        - weights.cvar_regret * tail_loss
        - weights.negative_regret * shortfall
    )


def metrics_table(
    per_day: pd.DataFrame,
    reference: pd.Series,
    blocks: list[pd.DatetimeIndex],
    milp: pd.Series | None = None,
    weights: SelectionWeights | None = None,
) -> pd.DataFrame:
    """Candidate-by-metric table; ``per_day`` has one column per candidate."""
    weights = weights if weights is not None else SelectionWeights()
    rows = {
        name: candidate_metrics(per_day[name], reference, blocks, milp, weights.cvar_alpha)
        for name in per_day.columns
    }
    table = pd.DataFrame.from_dict(rows, orient="index")
    table["selection_score"] = [selection_score(rows[name], weights) for name in table.index]
    return table


# ------------------------------------------------------------- selection rules


def _pareto_front(table: pd.DataFrame) -> list[str]:
    """Non-dominated candidates maximizing (mean, cvar_regret)."""
    front = []
    pts = table[["mean", "cvar_regret"]].to_numpy()
    for i, name in enumerate(table.index):
        dominated = ((pts >= pts[i]).all(axis=1) & (pts > pts[i]).any(axis=1)).any()
        if not dominated:
            front.append(name)
    return front


def select_pareto(table: pd.DataFrame) -> str:
    """Highest-median member of the (mean, downside) Pareto front."""
    return table.loc[_pareto_front(table), "median"].idxmax()


SELECTION_RULES: dict[str, callable] = {
    "highest_mean": lambda t: t["mean"].idxmax(),
    "highest_median": lambda t: t["median"].idxmax(),
    "highest_worst_fold": lambda t: t["worst_fold"].idxmax(),
    "risk_adjusted": lambda t: t["selection_score"].idxmax(),
    "lowest_downside": lambda t: t["downside_exposure"].idxmin(),
    "pareto": select_pareto,
}


# ------------------------------------------------------- leave-one-block-out


def leave_one_block_out(
    per_day: pd.DataFrame,
    reference: pd.Series,
    blocks: list[pd.DatetimeIndex],
    weights: SelectionWeights | None = None,
) -> pd.DataFrame:
    """Score every selection rule by out-of-block generalization.

    For each held-out block: rank candidates on the remaining blocks,
    apply each rule, then record the chosen candidate's mean revenue and
    regret on the held-out block, next to the block oracle (best
    candidate in hindsight) and the reference controller.
    """
    weights = weights if weights is not None else SelectionWeights()
    if len(blocks) < 2:
        raise ValueError("leave-one-block-out needs at least two blocks")
    records = []
    for held_idx, held in enumerate(blocks):
        inner_blocks = [b for i, b in enumerate(blocks) if i != held_idx]
        inner_days = per_day.index[~per_day.index.isin(held)]
        table = metrics_table(
            per_day.loc[inner_days], reference.loc[inner_days], inner_blocks, weights=weights
        )
        held_mask = per_day.index.isin(held)
        held_means = per_day.loc[held_mask].mean()
        ref_mean = float(reference.loc[held_mask].mean())
        oracle = held_means.idxmax()
        for rule, fn in SELECTION_RULES.items():
            chosen = fn(table)
            records.append(
                {
                    "held_block": held_idx,
                    "rule": rule,
                    "chosen": chosen,
                    "holdout_mean": float(held_means[chosen]),
                    "holdout_regret_mean": float(held_means[chosen] - ref_mean),
                    "oracle": oracle,
                    "oracle_mean": float(held_means[oracle]),
                    "gap_to_oracle": float(held_means[oracle] - held_means[chosen]),
                    "reference_mean": ref_mean,
                }
            )
    return pd.DataFrame.from_records(records)
