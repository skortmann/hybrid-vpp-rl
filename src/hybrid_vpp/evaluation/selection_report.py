"""Checkpoint-selection analysis report from the cached checkpoint matrix.

Reads the artifacts produced by
:mod:`hybrid_vpp.evaluation.checkpoint_matrix` and writes
``reports/checkpoint_selection_analysis.md``: seed-versus-checkpoint
variance decomposition, checkpoint-age effects, leave-one-block-out
selection-rule reliability, action-statistics correlations, and — last,
read-only, after the rules have been ranked on validation — the recorded
previous-phase test results of the five eval-best checkpoints.

Run as ``uv run python -m hybrid_vpp.evaluation.selection_report``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------- CONFIG

SPLIT = "val"
OUT_DIR = Path("artifacts/robust_selection")
CACHE_DIR = Path("artifacts/robust_selection/cache")
REPORT_PATH = Path("reports/checkpoint_selection_analysis.md")
TEST_EVALUATION = Path("artifacts/test_evaluation.json")  # frozen previous phase

# ------------------------------------------------------------------------


def _seed_label(candidate: str) -> tuple[int, str]:
    _, seed_part, label = candidate.split("_")
    return int(seed_part.removeprefix("seed")), label


def variance_decomposition(matrix: pd.DataFrame) -> dict[str, float]:
    """Two-way (seed x checkpoint) decomposition of validation means."""
    means = matrix["mean"].copy()
    idx = pd.MultiIndex.from_tuples([_seed_label(c) for c in means.index], names=["seed", "ckpt"])
    grid = means.set_axis(idx).unstack("ckpt")  # seeds x checkpoints
    grand = grid.to_numpy().mean()
    seed_eff = grid.mean(axis=1) - grand
    ckpt_eff = grid.mean(axis=0) - grand
    residual = grid.sub(seed_eff, axis=0).sub(ckpt_eff, axis=1) - grand
    return {
        "seed_std": float(seed_eff.std(ddof=1)),
        "checkpoint_std": float(ckpt_eff.std(ddof=1)),
        "interaction_std": float(residual.to_numpy().std(ddof=1)),
        "within_seed_spread": float(grid.std(axis=1, ddof=1).mean()),
        "within_checkpoint_spread": float(grid.std(axis=0, ddof=1).mean()),
        "grand_mean": float(grand),
    }


def action_statistics(split: str = SPLIT) -> pd.DataFrame:
    """Mean translated strategic parameters per checkpoint candidate."""
    rows = {}
    for path in sorted((CACHE_DIR / split).glob("ckpt_*.json")):
        data = json.loads(path.read_text())
        daa, ida, idc = [], [], []
        for rec in data.values():
            for a in rec["actions"].get("DAA_GATE_CLOSURE", []):
                daa.append(a[:2])
            for ev in ("IDA1_GATE_CLOSURE", "IDA2_GATE_CLOSURE", "IDA3_GATE_CLOSURE"):
                ida.extend(a[2] for a in rec["actions"].get(ev, []))
            idc.extend(a[3] for a in rec["actions"].get("IDC_DECISION", []))
        unit = lambda x: (np.asarray(x) + 1.0) / 2.0  # noqa: E731
        daa = np.asarray(daa)
        rows[path.stem] = {
            "daa_coverage": float(unit(daa[:, 0]).mean() * 1.2),
            "daa_arb_scale": float(unit(daa[:, 1]).mean()),
            "ida_gain": float(np.minimum(unit(ida) * 1.25, 1.25).mean()),
            "idc_gain": float(np.minimum(unit(idc) * 1.25, 1.25).mean()),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def main() -> None:
    matrix = pd.read_csv(OUT_DIR / "checkpoint_matrix.csv", index_col=0)
    lobo = pd.read_csv(OUT_DIR / "lobo_results.csv")
    payload = json.loads((OUT_DIR / "checkpoint_matrix.json").read_text())
    decomp = variance_decomposition(matrix)
    actions = action_statistics()

    # checkpoint-age effect and per-seed best checkpoint
    by = pd.MultiIndex.from_tuples([_seed_label(c) for c in matrix.index], names=["seed", "ckpt"])
    grid = matrix["mean"].set_axis(by).unstack("ckpt")
    age_means = grid.mean(axis=0).sort_index()
    best_per_seed = grid.idxmax(axis=1)

    # action statistics vs validation regret (Spearman)
    joined = actions.join(matrix[["mean_regret"]])
    corr = joined.corr(method="spearman")["mean_regret"].drop("mean_regret")

    lobo_summary = (
        lobo.groupby("rule")[["holdout_mean", "holdout_regret_mean", "gap_to_oracle"]]
        .agg(["mean", "min"])
        .round(0)
    )
    rule_rank = lobo.groupby("rule")["holdout_mean"].mean().sort_values(ascending=False)
    locked_rule = rule_rank.index[0]

    # read-only test read-out (previous phase, already recorded) — computed
    # AFTER the rules were ranked above; never feeds back into selection
    test = json.loads(TEST_EVALUATION.read_text())
    test_means = {f"ckpt_{s}_best": float(np.mean(v)) for s, v in test["rl_seeds"].items()}
    best_rows = matrix.loc[list(test_means)].assign(test_mean=pd.Series(test_means))
    val_test_corr = float(best_rows["mean"].corr(best_rows["test_mean"], method="spearman"))
    score_test_corr = float(
        best_rows["selection_score"].corr(best_rows["test_mean"], method="spearman")
    )

    lines = [
        "# Checkpoint-selection analysis (Phase 1)",
        "",
        f"92 validation days (2025-11-01 → 2026-01-31), {payload['n_blocks']} contiguous",
        f"blocks, {len(matrix)} candidates (5 seeds x 7 checkpoints). Baselines:",
        ", ".join(
            f"{k.removeprefix('baseline_')} {v:,.0f}" for k, v in payload["baseline_means"].items()
        )
        + " EUR/day.",
        "",
        "## Seed versus checkpoint variance",
        "",
        f"* seed effect std: **{decomp['seed_std']:,.0f} EUR/day**",
        f"* checkpoint effect std: **{decomp['checkpoint_std']:,.0f} EUR/day**",
        f"* seed x checkpoint interaction std: {decomp['interaction_std']:,.0f} EUR/day",
        f"* mean within-seed spread over checkpoints: {decomp['within_seed_spread']:,.0f}",
        f"* mean within-checkpoint spread over seeds: {decomp['within_checkpoint_spread']:,.0f}",
        "",
        "## Checkpoint age",
        "",
        "Mean validation revenue by training progress (EUR/day):",
        "",
        age_means.round(0).to_frame("mean_revenue").to_markdown(),
        "",
        "Best checkpoint per seed (full 92-day validation): "
        + ", ".join(f"seed{s}: {c}" for s, c in best_per_seed.items())
        + ".",
        "",
        "## Selection-rule reliability (leave-one-block-out)",
        "",
        "Mean/worst held-out-block performance of the candidate each rule picks:",
        "",
        lobo_summary.to_markdown(),
        "",
        f"Locked selection rule (highest mean held-out revenue): **{locked_rule}**.",
        "",
        "## Do action statistics predict validation regret?",
        "",
        "Spearman correlation of mean translated parameters with mean regret:",
        "",
        corr.round(2).to_frame("spearman_vs_regret").to_markdown(),
        "",
        "## Read-only test read-out (previous phase, five eval-best checkpoints)",
        "",
        "Recorded per-day test results from `artifacts/test_evaluation.json`;",
        "computed after the LOBO ranking above and never used for selection:",
        "",
        best_rows[["mean", "median", "selection_score", "test_mean"]].round(0).to_markdown(),
        "",
        f"Spearman(validation mean, test mean) over the five best checkpoints: "
        f"**{val_test_corr:+.2f}**; Spearman(risk-adjusted selection score, test mean): "
        f"**{score_test_corr:+.2f}**. The 30-day window used by the previous phase "
        f"ranked seed 0 first — the worst test generalizer; the blocked risk-adjusted "
        f"score ranks it last.",
        "",
    ]
    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines))
    print(f"wrote {REPORT_PATH}")
    print(f"locked rule: {locked_rule}; val-test corr (best ckpts): {val_test_corr:+.2f}")


if __name__ == "__main__":
    main()
