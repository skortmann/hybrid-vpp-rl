"""Benchmark-strengthening MILP variants on the validation split (Phase 6).

Evaluates the information-equivalent MILP with turnover penalties
(penalizing schedule churn across successive market re-optimizations)
and with renewable-forecast derating (a conservative/robust planning
variant) under identical information, gates, costs, and constraints.
Writes per-day caches next to the other candidates and a comparison
report ``reports/robust_milp_analysis.md``.

Run as ``uv run python -m hybrid_vpp.evaluation.milp_variants_eval``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------- CONFIG

SPLIT = "val"
N_BLOCKS = 6
CONFIG_PATH = Path("runs/experiments/V6-hybrid-sac-strategic-seed0.yaml")
VARIANTS = (
    {"name": "milp_turnover0.5", "turnover_penalty_eur_per_mwh": 0.5},
    {"name": "milp_turnover2", "turnover_penalty_eur_per_mwh": 2.0},
    {"name": "milp_derate0.95", "renewable_derate": 0.95},
    {"name": "milp_derate0.9", "renewable_derate": 0.9},
    # symmetric benchmark for the terminal-adjusted economics: no forced
    # end-of-day SoC, boundary priced by the adjusted metric instead
    {"name": "milp_no_terminal", "enforce_terminal_soc": False},
    # carry-over-compatible benchmark: terminal inventory valued at the
    # solve's own mean forecast price instead of constrained or free
    {
        "name": "milp_terminal_value",
        "enforce_terminal_soc": False,
        "terminal_value_from_prices": True,
    },
)
CACHE_DIR = Path("artifacts/robust_selection/cache")
REPORT_PATH = Path("reports/robust_milp_analysis.md")
N_PROCESSES = 4

# ------------------------------------------------------------------------


def run_variant(variant: dict, split: str = SPLIT) -> dict[str, dict]:
    from hybrid_vpp.config.models import load_config
    from hybrid_vpp.controllers.base import run_episode
    from hybrid_vpp.controllers.optimization import OptimizationController
    from hybrid_vpp.envs.hybrid_vpp_env import HybridVppEnv
    from hybrid_vpp.evaluation.checkpoint_matrix import METRIC_KEYS
    from hybrid_vpp.evaluation.metrics import episode_metrics
    from hybrid_vpp.evaluation.run_baselines import build_stack

    cfg = load_config(CONFIG_PATH)
    days = HybridVppEnv(cfg, split=split).valid_days
    _store, _profiles, sim, renewable_fc, price_fc = build_stack(cfg)
    kwargs = {k: v for k, v in variant.items() if k != "name"}
    controller = OptimizationController(cfg, renewable_fc, price_fc, **kwargs)

    out: dict[str, dict] = {}
    for day in days:
        run_episode(sim, controller, day)
        metrics = episode_metrics(sim)
        out[str(day.date())] = {"metrics": {k: metrics[k] for k in METRIC_KEYS if k in metrics}}
    return out


def _worker(variant: dict) -> str:
    import logging

    logging.basicConfig(level=logging.WARNING)
    result = run_variant(variant)
    path = CACHE_DIR / SPLIT / f"{variant['name']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result))
    tmp.replace(path)
    return variant["name"]


def report() -> None:
    from hybrid_vpp.evaluation.blocked_validation import (
        SelectionWeights,
        contiguous_blocks,
        metrics_table,
    )
    from hybrid_vpp.evaluation.checkpoint_matrix import load_per_day_table

    table = load_per_day_table(SPLIT)
    variants = ["baseline_milp_info"] + [v["name"] for v in VARIANTS if v["name"] in table.columns]
    blocks = contiguous_blocks(table.index, N_BLOCKS)
    stats = metrics_table(
        table[variants + ["ensemble_mean", "gate_c_r0.1"]],
        table["baseline_rule_based"],
        blocks,
        weights=SelectionWeights(),
    ).sort_values("mean", ascending=False)
    show = ["mean", "median", "cvar_regret", "mean_regret", "downside_exposure"]
    lines = [
        "# Robust MILP benchmark variants (Phase 6)",
        "",
        "All variants use the same information set, gate closures,",
        "transaction costs, deviation penalties, and physical constraints",
        "as the classic information-equivalent MILP; only the planning",
        "objective changes. RL composites shown for reference. Regret is",
        "vs the rule-based controller on the same 92 validation days.",
        "",
        stats[show].round(0).to_markdown(),
        "",
        "Turnover penalties price the churn between successive market",
        "re-optimizations; forecast derating plans against a conservative",
        "renewable estimate (solver: Gurobi, 30 s limit per solve).",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines))
    print(f"wrote {REPORT_PATH}")
    print(stats[show].round(0).to_string())


def main() -> None:
    import multiprocessing as mp

    todo = [v for v in VARIANTS if not (CACHE_DIR / SPLIT / f"{v['name']}.json").exists()]
    if todo:
        print(f"evaluating {len(todo)} MILP variants with {N_PROCESSES} processes")
        with mp.get_context("spawn").Pool(min(N_PROCESSES, len(todo))) as pool:
            for name in pool.imap_unordered(_worker, todo):
                data = json.loads((CACHE_DIR / SPLIT / f"{name}.json").read_text())
                rev = [r["metrics"]["total_net_revenue_eur"] for r in data.values()]
                print(
                    f"  {name}: mean {np.mean(rev):,.0f} median {np.median(rev):,.0f}", flush=True
                )
    report()


if __name__ == "__main__":
    main()
