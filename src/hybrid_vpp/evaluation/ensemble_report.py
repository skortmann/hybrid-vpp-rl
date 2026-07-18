"""Ensemble and disagreement analysis report (Phase 2).

Compares the action-space ensembles against the individual member
policies and the baselines on blocked validation folds, quantifies the
seed-variance reduction, and tests whether ensemble disagreement
predicts poor RL performance (the precondition for using disagreement
as a safety signal). Writes ``reports/ensemble_analysis.md`` plus
``artifacts/robust_selection/ensemble_results.json`` and
``disagreement_analysis.json``.

Run as ``uv run python -m hybrid_vpp.evaluation.ensemble_report`` after
``hybrid_vpp.evaluation.ensemble_eval``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------- CONFIG

SPLIT = "val"
N_BLOCKS = 6
MEMBER_LABEL = "best"
OUT_DIR = Path("artifacts/robust_selection")
CACHE_DIR = Path("artifacts/robust_selection/cache")
REPORT_PATH = Path("reports/ensemble_analysis.md")

# ------------------------------------------------------------------------


def disagreement_frame(split: str = SPLIT) -> pd.DataFrame:
    """Per-day disagreement of the mean ensemble with member/RB regret."""
    from hybrid_vpp.evaluation.checkpoint_matrix import load_per_day_table

    data = json.loads((CACHE_DIR / split / "ensemble_mean.json").read_text())
    table = load_per_day_table(split)
    members = [c for c in table.columns if c.startswith("ckpt_") and c.endswith(f"_{MEMBER_LABEL}")]
    rows = {}
    for day, rec in data.items():
        ts = pd.Timestamp(day)
        rows[ts] = {
            "u_market_mean": rec["u_market_mean"],
            "u_market_max": rec["u_market_max"],
            "ensemble_regret": table.at[ts, "ensemble_mean"] - table.at[ts, "baseline_rule_based"],
            "member_mean_regret": float(
                table.loc[ts, members].mean() - table.at[ts, "baseline_rule_based"]
            ),
            "abs_deviation_mwh": rec["metrics"]["abs_deviation_mwh"],
            "imbalance_cash_eur": rec["metrics"]["imbalance_cash_eur"],
        }
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


def main() -> None:
    from hybrid_vpp.evaluation.blocked_validation import (
        SelectionWeights,
        contiguous_blocks,
        metrics_table,
    )
    from hybrid_vpp.evaluation.checkpoint_matrix import load_per_day_table

    table = load_per_day_table(SPLIT)
    members = [c for c in table.columns if c.startswith("ckpt_") and c.endswith(f"_{MEMBER_LABEL}")]
    ensembles = [c for c in table.columns if c.startswith("ensemble_")]
    reference = table["baseline_rule_based"]
    blocks = contiguous_blocks(table.index, N_BLOCKS)
    stats = metrics_table(
        table[ensembles + members],
        reference,
        blocks,
        milp=table.get("baseline_milp_info"),
        weights=SelectionWeights(),
    ).sort_values("mean", ascending=False)

    member_means = table[members].mean()
    dis = disagreement_frame(SPLIT)
    high_u = dis["u_market_mean"] > dis["u_market_mean"].median()
    dis_summary = {
        "spearman_u_vs_ensemble_regret": float(
            dis["u_market_mean"].corr(dis["ensemble_regret"], method="spearman")
        ),
        "spearman_u_vs_member_mean_regret": float(
            dis["u_market_mean"].corr(dis["member_mean_regret"], method="spearman")
        ),
        "spearman_u_vs_abs_deviation": float(
            dis["u_market_mean"].corr(dis["abs_deviation_mwh"], method="spearman")
        ),
        "mean_regret_high_disagreement": float(dis.loc[high_u, "ensemble_regret"].mean()),
        "mean_regret_low_disagreement": float(dis.loc[~high_u, "ensemble_regret"].mean()),
        "p_negative_regret_high_u": float((dis.loc[high_u, "ensemble_regret"] < 0).mean()),
        "p_negative_regret_low_u": float((dis.loc[~high_u, "ensemble_regret"] < 0).mean()),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "ensemble_results.json").write_text(
        json.dumps(
            {
                "split": SPLIT,
                "members": members,
                "member_mean_range": [float(member_means.min()), float(member_means.max())],
                "member_mean_std": float(member_means.std(ddof=1)),
                "stats": json.loads(stats.to_json(orient="index")),
            },
            indent=2,
        )
    )
    (OUT_DIR / "disagreement_analysis.json").write_text(json.dumps(dis_summary, indent=2))

    show = [
        "mean",
        "median",
        "fold_std",
        "worst_fold",
        "cvar_regret",
        "mean_regret",
        "p_beat_reference",
        "downside_exposure",
    ]
    lines = [
        "# Ensemble analysis (Phase 2)",
        "",
        f"Members: the five eval-best checkpoints (validation means "
        f"{member_means.min():,.0f} – {member_means.max():,.0f}, std "
        f"{member_means.std(ddof=1):,.0f} EUR/day). Rule-based reference: "
        f"{reference.mean():,.0f} EUR/day. 92 validation days, {N_BLOCKS} blocks.",
        "",
        "## Ensembles versus members (blocked validation)",
        "",
        stats[show].round({c: 2 if c == "p_beat_reference" else 0 for c in show}).to_markdown(),
        "",
        "An ensemble is one deterministic controller: it removes the seed-",
        "selection step entirely. Compare its row against the spread of the",
        "member rows to judge the variance reduction.",
        "",
        "## Does disagreement predict poor performance?",
        "",
        f"* Spearman(u, ensemble daily regret): "
        f"**{dis_summary['spearman_u_vs_ensemble_regret']:+.2f}**",
        f"* Spearman(u, member-mean daily regret): "
        f"{dis_summary['spearman_u_vs_member_mean_regret']:+.2f}",
        f"* Spearman(u, absolute deviation): {dis_summary['spearman_u_vs_abs_deviation']:+.2f}",
        f"* mean regret on high-disagreement days (above median u): "
        f"{dis_summary['mean_regret_high_disagreement']:+,.0f} EUR/day",
        f"* mean regret on low-disagreement days: "
        f"{dis_summary['mean_regret_low_disagreement']:+,.0f} EUR/day",
        f"* P(negative regret | high u): {dis_summary['p_negative_regret_high_u']:.0%}; "
        f"P(negative regret | low u): {dis_summary['p_negative_regret_low_u']:.0%}",
        "",
    ]
    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines))
    print(f"wrote {REPORT_PATH}")
    print(stats[show].round({c: 2 if c == "p_beat_reference" else 0 for c in show}).to_string())
    print(json.dumps(dis_summary, indent=1))


if __name__ == "__main__":
    main()
