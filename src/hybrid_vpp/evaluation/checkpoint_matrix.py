"""Checkpoint-matrix evaluation on blocked temporal validation folds.

Rolls out every saved checkpoint of every training seed (and the
baseline controllers) over all validation days, caching per-day episode
metrics and raw strategic actions per market event. The cached table is
then analysed with :mod:`hybrid_vpp.evaluation.blocked_validation`:
fold metrics, the predefined selection rules, and leave-one-block-out
selection reliability.

Run as ``uv run python -m hybrid_vpp.evaluation.checkpoint_matrix``;
tunables live in the CONFIG block below. Results are cached per
(checkpoint, split) so re-runs are incremental.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------- CONFIG

SPLIT = "val"
N_BLOCKS = 6
CONFIG_PATH = Path("runs/experiments/V6-hybrid-sac-strategic-seed0.yaml")
RUN_GLOB = "V6-hybrid-sac-strategic-seed{seed}_2*"
SEEDS = (0, 1, 2, 3, 4)
BASELINES = ("do_nothing", "rule_based", "milp_info")
CACHE_DIR = Path("artifacts/robust_selection/cache")
OUT_DIR = Path("artifacts/robust_selection")
N_PROCESSES = 6

# ------------------------------------------------------------------------


def list_checkpoints(seeds: tuple[int, ...] = SEEDS) -> list[dict]:
    """All saved checkpoints per seed: step checkpoints plus the eval-best."""
    refs = []
    for seed in seeds:
        run_dir = sorted(Path("runs").glob(RUN_GLOB.format(seed=seed)))[-1]
        for ckpt in sorted(run_dir.glob("checkpoints/model_*_steps.zip")):
            steps = int(ckpt.stem.split("_")[1])
            refs.append({"seed": seed, "label": f"{steps // 1000:03d}k", "path": str(ckpt)})
        refs.append({"seed": seed, "label": "best", "path": str(run_dir / "best/best_model.zip")})
    return refs


def _cache_path(name: str, split: str) -> Path:
    return CACHE_DIR / split / f"{name}.json"


METRIC_KEYS = (
    "total_net_revenue_eur",
    "terminal_energy_value_eur",
    "total_net_revenue_terminal_adjusted_eur",
    "market_revenue_eur",
    "imbalance_cash_eur",
    "abs_deviation_mwh",
    "equivalent_full_cycles",
    "transaction_cost_eur",
    "daa_volume_mwh",
    "grid_export_mwh",
    "curtailment_ratio",
)


def evaluate_policy_on_split(
    checkpoint_path: str | Path, split: str = SPLIT, record_actions: bool = True
) -> dict[str, dict]:
    """Deterministic rollout of one checkpoint over all days of ``split``."""
    import torch

    from hybrid_vpp.config.models import load_config
    from hybrid_vpp.envs.hybrid_vpp_env import HybridVppEnv
    from hybrid_vpp.training.algorithms import algo_class

    torch.set_num_threads(1)
    cfg = load_config(CONFIG_PATH)
    env = HybridVppEnv(cfg, split=split)
    model = algo_class("sac").load(checkpoint_path)

    out: dict[str, dict] = {}
    for day in env.valid_days:
        obs, info = env.reset(options={"day": day})
        actions: dict[str, list[list[float]]] = {}
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            if record_actions:
                actions.setdefault(info["event_type"], []).append(
                    [round(float(a), 5) for a in action]
                )
            obs, _, done, _, info = env.step(action)
        metrics = info["episode_metrics"]
        out[str(day.date())] = {
            "metrics": {k: metrics[k] for k in METRIC_KEYS if k in metrics},
            "actions": actions if record_actions else {},
        }
    return out


def evaluate_baseline_on_split(name: str, split: str = SPLIT) -> dict[str, dict]:
    """Per-day metrics for one baseline controller on ``split``."""
    from hybrid_vpp.config.models import load_config
    from hybrid_vpp.controllers.base import run_episode
    from hybrid_vpp.controllers.rule_based import RuleBasedController
    from hybrid_vpp.controllers.simple import DoNothingController
    from hybrid_vpp.envs.hybrid_vpp_env import HybridVppEnv
    from hybrid_vpp.evaluation.metrics import episode_metrics
    from hybrid_vpp.evaluation.run_baselines import build_stack

    cfg = load_config(CONFIG_PATH)
    days = HybridVppEnv(cfg, split=split).valid_days
    store, profiles, sim, renewable_fc, price_fc = build_stack(cfg)
    if name == "do_nothing":
        controller = DoNothingController(cfg, renewable_fc)
    elif name == "rule_based":
        controller = RuleBasedController(cfg, renewable_fc, price_fc)
    elif name == "milp_info":
        from hybrid_vpp.controllers.optimization import OptimizationController

        controller = OptimizationController(cfg, renewable_fc, price_fc)
    else:
        raise ValueError(f"unknown baseline: {name}")

    out: dict[str, dict] = {}
    for day in days:
        run_episode(sim, controller, day)
        metrics = episode_metrics(sim)
        out[str(day.date())] = {"metrics": {k: metrics[k] for k in METRIC_KEYS if k in metrics}}
    return out


def _worker(task: dict) -> str:
    """Evaluate one checkpoint or baseline and write its cache file."""
    import logging

    logging.basicConfig(level=logging.WARNING)
    if task["kind"] == "checkpoint":
        result = evaluate_policy_on_split(task["path"], task["split"])
    else:
        result = evaluate_baseline_on_split(task["name"], task["split"])
    path = _cache_path(task["cache_name"], task["split"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result))
    tmp.replace(path)
    return task["cache_name"]


def run_evaluations(split: str = SPLIT, n_processes: int = N_PROCESSES) -> None:
    """Evaluate all uncached checkpoints and baselines (spawn, one per task)."""
    import multiprocessing as mp

    tasks = []
    for ref in list_checkpoints():
        name = f"ckpt_seed{ref['seed']}_{ref['label']}"
        if not _cache_path(name, split).exists():
            tasks.append({"kind": "checkpoint", "cache_name": name, "split": split, **ref})
    for baseline in BASELINES:
        name = f"baseline_{baseline}"
        if not _cache_path(name, split).exists():
            tasks.append({"kind": "baseline", "cache_name": name, "split": split, "name": baseline})
    if not tasks:
        print("all evaluations cached")
        return
    print(f"evaluating {len(tasks)} tasks on split={split} with {n_processes} processes")
    with mp.get_context("spawn").Pool(n_processes) as pool:
        for done in pool.imap_unordered(_worker, tasks):
            print(f"  done: {done}", flush=True)


def load_per_day_table(split: str = SPLIT) -> pd.DataFrame:
    """Days x candidates revenue table from the cache (baselines included)."""
    columns = {}
    for path in sorted((CACHE_DIR / split).glob("*.json")):
        data = json.loads(path.read_text())
        columns[path.stem] = {
            pd.Timestamp(day): rec["metrics"]["total_net_revenue_eur"] for day, rec in data.items()
        }
    table = pd.DataFrame(columns).sort_index()
    if table.isna().any().any():
        missing = table.columns[table.isna().any()].tolist()
        raise RuntimeError(f"incomplete caches (day mismatch): {missing}")
    return table


def analyse(split: str = SPLIT, n_blocks: int = N_BLOCKS) -> dict:
    """Fold metrics, selection rules, and LOBO reliability from the cache."""
    import pandas as pd

    from hybrid_vpp.evaluation.blocked_validation import (
        SELECTION_RULES,
        SelectionWeights,
        contiguous_blocks,
        leave_one_block_out,
        metrics_table,
    )

    table = load_per_day_table(split)
    candidates = table[[c for c in table.columns if c.startswith("ckpt_")]]
    reference = table["baseline_rule_based"]
    milp = table.get("baseline_milp_info")
    blocks = contiguous_blocks(table.index, n_blocks)
    weights = SelectionWeights()

    matrix = metrics_table(candidates, reference, blocks, milp=milp, weights=weights)
    fold_rows = []
    for fold_idx, block in enumerate(blocks):
        sub = table.loc[table.index.isin(block)]
        for cand in table.columns:
            fold_rows.append(
                {
                    "candidate": cand,
                    "fold": fold_idx,
                    "fold_start": str(block[0].date()),
                    "fold_end": str(block[-1].date()),
                    "mean": float(sub[cand].mean()),
                    "median": float(sub[cand].median()),
                }
            )
    folds = pd.DataFrame.from_records(fold_rows)
    lobo = leave_one_block_out(candidates, reference, blocks, weights)
    selected = {rule: fn(matrix) for rule, fn in SELECTION_RULES.items()}
    lobo_summary = (
        lobo.groupby("rule")[["holdout_mean", "holdout_regret_mean", "gap_to_oracle"]]
        .mean()
        .sort_values("holdout_mean", ascending=False)
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / f"per_day_{split}.csv")
    matrix.to_csv(OUT_DIR / "checkpoint_matrix.csv")
    folds.to_csv(OUT_DIR / "fold_results.csv", index=False)
    lobo.to_csv(OUT_DIR / "lobo_results.csv", index=False)
    payload = {
        "split": split,
        "n_blocks": n_blocks,
        "weights": vars(weights),
        "selected_by_rule": selected,
        "lobo_summary": {k: v.to_dict() for k, v in lobo_summary.iterrows()},
        "baseline_means": {
            c: float(table[c].mean()) for c in table.columns if c.startswith("baseline_")
        },
        "matrix": json.loads(matrix.to_json(orient="index")),
    }
    (OUT_DIR / "checkpoint_matrix.json").write_text(json.dumps(payload, indent=2))
    print(f"candidates: {len(matrix)}; selected by rule: {selected}")
    print(lobo_summary)
    return payload


def main() -> None:
    run_evaluations()
    analyse()


if __name__ == "__main__":
    main()
