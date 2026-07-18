"""Plots for the robustness/selection research phase.

Renders the standard figures from the cached robust-selection artifacts
into ``docs/assets/robust/`` so the MkDocs pages can embed them. Every
figure is regenerated from artifacts only — no rollouts happen here.

Run as ``uv run python -m hybrid_vpp.evaluation.robust_plots``. Figures
that lack their inputs are skipped with a note, so the module can be
re-run after every phase.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# --------------------------------------------------------------------- CONFIG

OUT_DIR = Path("artifacts/robust_selection")
CACHE_DIR = Path("artifacts/robust_selection/cache")
FIG_DIR = Path("docs/assets/robust")
SPLIT = "val"

# ------------------------------------------------------------------------


def _save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=150)
    plt.close(fig)
    print(f"wrote {FIG_DIR / name}")


def plot_fold_performance() -> None:
    folds = pd.read_csv(OUT_DIR / "fold_results.csv")
    ckpt = folds[folds.candidate.str.startswith("ckpt_")]
    rb = folds[folds.candidate == "baseline_rule_based"].set_index("fold")["mean"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for _cand, sub in ckpt.groupby("candidate"):
        ax.plot(sub["fold"], sub["mean"], color="steelblue", alpha=0.25, lw=1)
    ax.plot(rb.index, rb.values, color="black", lw=2, label="rule-based")
    milp = folds[folds.candidate == "baseline_milp_info"].set_index("fold")["mean"]
    ax.plot(milp.index, milp.values, color="darkred", lw=2, ls="--", label="info-MILP")
    ax.set_xlabel("validation block")
    ax.set_ylabel("mean revenue [EUR/day]")
    ax.set_title("Fold-level performance: 35 checkpoints vs baselines")
    ax.legend()
    _save(fig, "fold_performance.png")


def plot_seed_checkpoint_dispersion() -> None:
    matrix = pd.read_csv(OUT_DIR / "checkpoint_matrix.csv", index_col=0)
    ckpt = matrix[matrix.index.str.startswith("ckpt_")].copy()
    parts = ckpt.index.str.split("_")
    ckpt["seed"] = [p[1] for p in parts]
    ckpt["label"] = [p[2] for p in parts]
    grid = ckpt.pivot(index="seed", columns="label", values="mean")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(grid.to_numpy(), aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(len(grid.columns)), grid.columns)
    ax.set_yticks(range(len(grid.index)), grid.index)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f"{grid.iat[i, j] / 1000:.1f}k", ha="center", va="center", fontsize=8)
    fig.colorbar(im, label="mean validation revenue [EUR/day]")
    ax.set_title("Seed x checkpoint validation means")
    _save(fig, "seed_checkpoint_matrix.png")


def plot_ensemble_comparison() -> None:
    path = OUT_DIR / "ensemble_results.json"
    if not path.exists():
        print("skip ensemble_comparison (no ensemble_results.json yet)")
        return
    stats = pd.DataFrame.from_dict(json.loads(path.read_text())["stats"], orient="index")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["darkorange" if i.startswith("ensemble") else "steelblue" for i in stats.index]
    ax.barh(stats.index, stats["mean_regret"], color=colors)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("mean daily regret vs rule-based [EUR/day]")
    ax.set_title("Ensembles (orange) vs member policies (blue)")
    _save(fig, "ensemble_comparison.png")


def plot_disagreement_vs_regret() -> None:
    path = CACHE_DIR / SPLIT / "ensemble_mean.json"
    if not path.exists():
        print("skip disagreement_vs_regret (no ensemble cache yet)")
        return
    from hybrid_vpp.evaluation.ensemble_report import disagreement_frame

    dis = disagreement_frame(SPLIT)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(dis["u_market_mean"], dis["ensemble_regret"], s=18, alpha=0.7)
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("mean market-event disagreement u")
    ax.set_ylabel("ensemble daily regret vs rule-based [EUR]")
    ax.set_title("Policy disagreement vs daily regret (validation)")
    _save(fig, "disagreement_vs_regret.png")


def plot_daily_rl_vs_rule_based() -> None:
    per_day = pd.read_csv(OUT_DIR / f"per_day_{SPLIT}.csv", index_col=0, parse_dates=True)
    col = "ensemble_mean" if "ensemble_mean" in per_day.columns else None
    if col is None:
        print("skip daily_rl_vs_rule_based (no ensemble column yet)")
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    regret = per_day[col] - per_day["baseline_rule_based"]
    ax.bar(
        per_day.index,
        regret,
        width=1.0,
        color=["tab:green" if r > 0 else "tab:red" for r in regret],
    )
    ax.set_ylabel("ensemble minus rule-based [EUR/day]")
    ax.set_title("Daily paired regret of the mean ensemble (validation)")
    _save(fig, "daily_regret.png")


#: headline controllers of the validation comparison: column -> (label, color)
COMPARISON = {
    "baseline_rule_based": ("rule-based", "black"),
    "baseline_milp_info": ("info-equivalent MILP", "darkred"),
    "ensemble_mean": ("RL ensemble (ungated)", "tab:orange"),
    "gate_c_r0.1": ("RL ensemble + residual bound (promoted)", "tab:green"),
}


def _comparison_table() -> pd.DataFrame | None:
    per_day = pd.read_csv(OUT_DIR / f"per_day_{SPLIT}.csv", index_col=0, parse_dates=True)
    missing = [c for c in COMPARISON if c not in per_day.columns]
    if missing:
        print(f"skip validation comparison (missing columns: {missing})")
        return None
    return per_day


def plot_val_rolling_revenue() -> None:
    """7-day rolling mean of daily revenue, with the per-seed RL range."""
    per_day = _comparison_table()
    if per_day is None:
        return
    fig, ax = plt.subplots(figsize=(9, 4.8))
    seeds = [c for c in per_day.columns if c.startswith("ckpt_") and c.endswith("_best")]
    band = per_day[seeds].rolling(7, min_periods=3)
    ax.fill_between(
        per_day.index,
        band.mean().min(axis=1),
        band.mean().max(axis=1),
        color="steelblue",
        alpha=0.18,
        label="single-seed RL range (5 best checkpoints)",
    )
    for col, (label, color) in COMPARISON.items():
        ls = "--" if col == "baseline_milp_info" else "-"
        ax.plot(
            per_day.index,
            per_day[col].rolling(7, min_periods=3).mean(),
            color=color,
            ls=ls,
            lw=1.8,
            label=label,
        )
    ax.set_ylabel("7-day rolling mean revenue [EUR/day]")
    ax.set_title("Validation split (92 days): controller revenue")
    ax.legend(fontsize=8, loc="lower left")
    _save(fig, "val_rolling_revenue.png")


def plot_val_cumulative_vs_rule_based() -> None:
    """Cumulative paired difference to rule-based over the validation split."""
    per_day = _comparison_table()
    if per_day is None:
        return
    ref = per_day["baseline_rule_based"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for col, (label, color) in COMPARISON.items():
        if col == "baseline_rule_based":
            continue
        ax.plot(
            per_day.index,
            (per_day[col] - ref).cumsum() / 1000.0,
            color=color,
            lw=1.8,
            label=label,
        )
    ax.axhline(0, color="black", lw=1)
    ax.set_ylabel("cumulative difference to rule-based [kEUR]")
    ax.set_title("Validation split: cumulative paired difference vs rule-based")
    ax.legend(fontsize=8)
    _save(fig, "val_cumulative_vs_rule_based.png")


def plot_val_regret_distribution() -> None:
    """Distribution of daily paired regret vs rule-based (the tail story)."""
    per_day = _comparison_table()
    if per_day is None:
        return
    ref = per_day["baseline_rule_based"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    candidates = [c for c in COMPARISON if c != "baseline_rule_based"]
    # left: empirical CDF of daily regret
    for col in candidates:
        label, color = COMPARISON[col]
        regret = (per_day[col] - ref).sort_values()
        axes[0].step(
            regret, (pd.RangeIndex(1, len(regret) + 1)) / len(regret), color=color, label=label
        )
    axes[0].axvline(0, color="black", lw=1)
    axes[0].set_xlabel("daily regret vs rule-based [EUR]")
    axes[0].set_ylabel("fraction of days")
    axes[0].set_title("Empirical CDF")
    axes[0].legend(fontsize=7, loc="upper left")
    # right: boxplot on the same values
    data = [(per_day[c] - ref).to_numpy() for c in candidates]
    box = axes[1].boxplot(
        data, tick_labels=[COMPARISON[c][0].replace(" (", "\n(") for c in candidates], widths=0.55
    )
    for median in box["medians"]:
        median.set_color("black")
    axes[1].axhline(0, color="black", lw=1)
    axes[1].set_ylabel("daily regret vs rule-based [EUR]")
    axes[1].set_title("Distribution (whiskers: 1.5 IQR)")
    axes[1].tick_params(axis="x", labelsize=7)
    fig.suptitle("Validation split: daily paired regret vs rule-based", fontsize=11)
    _save(fig, "val_regret_distribution.png")


def plot_val_scatter_vs_rule_based() -> None:
    """Day-level pairing: each controller against rule-based revenue."""
    per_day = _comparison_table()
    if per_day is None:
        return
    ref = per_day["baseline_rule_based"]
    candidates = [c for c in COMPARISON if c != "baseline_rule_based"]
    fig, axes = plt.subplots(1, len(candidates), figsize=(11, 3.9), sharex=True, sharey=True)
    lims = (
        min(per_day[list(COMPARISON)].min()) / 1000.0,
        max(per_day[list(COMPARISON)].max()) / 1000.0,
    )
    for ax, col in zip(axes, candidates, strict=True):
        label, color = COMPARISON[col]
        ax.plot(lims, lims, color="grey", lw=1, ls=":")
        ax.scatter(ref / 1000.0, per_day[col] / 1000.0, s=14, alpha=0.65, color=color)
        wins = float((per_day[col] > ref).mean())
        ax.set_title(f"{label}\ndays above rule-based: {wins:.0%}", fontsize=9)
        ax.set_xlabel("rule-based [kEUR/day]")
    axes[0].set_ylabel("controller [kEUR/day]")
    fig.suptitle("Validation split: day-level pairing against rule-based", fontsize=11)
    _save(fig, "val_scatter_vs_rule_based.png")


def main() -> None:
    plot_fold_performance()
    plot_seed_checkpoint_dispersion()
    plot_ensemble_comparison()
    plot_disagreement_vs_regret()
    plot_daily_rl_vs_rule_based()
    plot_val_rolling_revenue()
    plot_val_cumulative_vs_rule_based()
    plot_val_regret_distribution()
    plot_val_scatter_vs_rule_based()


if __name__ == "__main__":
    main()
