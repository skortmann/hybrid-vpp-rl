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
    for cand, sub in ckpt.groupby("candidate"):
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


def main() -> None:
    plot_fold_performance()
    plot_seed_checkpoint_dispersion()
    plot_ensemble_comparison()
    plot_disagreement_vs_regret()
    plot_daily_rl_vs_rule_based()


if __name__ == "__main__":
    main()
