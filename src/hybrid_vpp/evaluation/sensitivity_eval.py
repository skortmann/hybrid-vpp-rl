"""Zero-shot sensitivity of the promoted controller (Phase 7).

Evaluates the locked candidate (mean ensemble + bounded residual 0.1 +
deterministic dispatch) and the rule-based reference under perturbed
environment assumptions, without retraining — the policies were trained
on the base configuration, so every scenario measures zero-shot
robustness, stated as such.

Two mechanisms:

* **Analytic deviation-penalty sweep** — settlement applies
  ``-p * |deviation|`` linearly, so zero-shot revenue under penalty p is
  ``revenue(25) + (25 - p) * deviation`` per day, computed exactly from
  the cached rollouts for every candidate.
* **Rollout scenarios** — grid export limit, BESS energy capacity, and
  renewable forecast quality change the physics and must be re-rolled.

Run as ``uv run python -m hybrid_vpp.evaluation.sensitivity_eval``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------- CONFIG

SPLIT = "val"
CONFIG_PATH = Path("runs/experiments/V6-hybrid-sac-strategic-seed0.yaml")
RESIDUAL_BOUND = 0.1  # the locked gate
PENALTIES = (0.0, 10.0, 25.0, 50.0, 100.0)
SWEEP_CANDIDATES = ("gate_c_r0.1", "ensemble_mean", "baseline_rule_based", "baseline_milp_info")
CACHE_DIR = Path("artifacts/robust_selection/cache")
SENS_CACHE = Path("artifacts/robust_selection/cache/val_sensitivity")
OUT_PATH = Path("artifacts/robust_selection/sensitivity_results.json")
N_PROCESSES = 6

# ------------------------------------------------------------------------


def scenario_list() -> list[dict]:
    return [
        {"name": "export_limit_x0.8", "export_limit_factor": 0.8},
        {"name": "export_limit_x1.2", "export_limit_factor": 1.2},
        {"name": "bess_energy_x0.5", "bess_energy_factor": 0.5},
        {"name": "bess_energy_x2", "bess_energy_factor": 2.0},
        {"name": "forecast_perfect", "renewable_mode": "perfect"},
        {"name": "forecast_persistence", "renewable_mode": "persistence"},
    ]


def _apply_scenario(cfg, scenario: dict) -> None:
    if "export_limit_factor" in scenario:
        cfg.site.grid.export_limit_mw *= scenario["export_limit_factor"]
    if "bess_energy_factor" in scenario:
        cfg.site.battery.energy_capacity_mwh *= scenario["bess_energy_factor"]
    if "renewable_mode" in scenario:
        cfg.forecast.renewable_mode = scenario["renewable_mode"]


def penalty_sweep() -> dict:
    """Exact zero-shot revenue under every penalty, from cached rollouts."""
    out: dict[str, dict] = {}
    for cand in SWEEP_CANDIDATES:
        cache = CACHE_DIR / SPLIT / f"{cand}.json"
        data = json.loads(cache.read_text())
        revenue = np.array([r["metrics"]["total_net_revenue_eur"] for r in data.values()])
        deviation = np.array([r["metrics"]["abs_deviation_mwh"] for r in data.values()])
        out[cand] = {
            f"penalty_{p:g}": {
                "mean": float((revenue + (25.0 - p) * deviation).mean()),
                "median": float(np.median(revenue + (25.0 - p) * deviation)),
            }
            for p in PENALTIES
        }
        out[cand]["mean_abs_deviation_mwh"] = float(deviation.mean())
    return out


def run_scenario(task: dict) -> dict[str, dict]:
    """Roll one (scenario, controller) pair over the validation days."""
    import torch

    from hybrid_vpp.config.models import load_config
    from hybrid_vpp.envs.ensemble import PolicyEnsemble
    from hybrid_vpp.envs.hybrid_vpp_env import HybridVppEnv
    from hybrid_vpp.envs.safety_gate import SafetyGate, rule_equivalent_action
    from hybrid_vpp.evaluation.checkpoint_matrix import METRIC_KEYS
    from hybrid_vpp.evaluation.ensemble_eval import member_paths
    from hybrid_vpp.training.algorithms import algo_class

    torch.set_num_threads(1)
    cfg = load_config(CONFIG_PATH)
    _apply_scenario(cfg, task["scenario"])
    env = HybridVppEnv(cfg, split=SPLIT)

    out: dict[str, dict] = {}
    if task["controller"] == "rule_based":
        from hybrid_vpp.controllers.base import run_episode
        from hybrid_vpp.controllers.rule_based import RuleBasedController
        from hybrid_vpp.evaluation.metrics import episode_metrics
        from hybrid_vpp.evaluation.run_baselines import build_stack

        _s, _p, sim, renewable_fc, price_fc = build_stack(cfg)
        controller = RuleBasedController(cfg, renewable_fc, price_fc)
        for day in env.valid_days:
            run_episode(sim, controller, day)
            metrics = episode_metrics(sim)
            out[str(day.date())] = {"metrics": {k: metrics[k] for k in METRIC_KEYS if k in metrics}}
        return out

    models = [algo_class("sac").load(p) for p in member_paths()]
    ensemble = PolicyEnsemble(models, mode="mean")
    gate = SafetyGate(
        "bounded_residual",
        rule_equivalent_action(cfg.episode.strategic_gain_max),
        max_residual=RESIDUAL_BOUND,
    )
    for day in env.valid_days:
        obs, info = env.reset(options={"day": day})
        done = False
        while not done:
            proposal, _ = ensemble.predict(obs)
            action, _rec = gate.apply(proposal, info["event_type"])
            obs, _, done, _, info = env.step(action)
        metrics = info["episode_metrics"]
        out[str(day.date())] = {"metrics": {k: metrics[k] for k in METRIC_KEYS if k in metrics}}
    return out


def _worker(task: dict) -> str:
    import logging

    logging.basicConfig(level=logging.WARNING)
    name = f"{task['scenario']['name']}__{task['controller']}"
    result = run_scenario(task)
    SENS_CACHE.mkdir(parents=True, exist_ok=True)
    tmp = SENS_CACHE / f"{name}.tmp"
    tmp.write_text(json.dumps(result))
    tmp.replace(SENS_CACHE / f"{name}.json")
    return name


def main() -> None:
    import multiprocessing as mp

    tasks = [
        {"scenario": s, "controller": c}
        for s in scenario_list()
        for c in ("gate_c", "rule_based")
        if not (SENS_CACHE / f"{s['name']}__{c}.json").exists()
    ]
    if tasks:
        print(f"evaluating {len(tasks)} sensitivity rollouts with {N_PROCESSES} processes")
        with mp.get_context("spawn").Pool(min(N_PROCESSES, len(tasks))) as pool:
            for name in pool.imap_unordered(_worker, tasks):
                print(f"  done: {name}", flush=True)

    matrix: dict[str, dict] = {"penalty_sweep_zero_shot": penalty_sweep(), "scenarios": {}}
    for s in scenario_list():
        row = {}
        for c in ("gate_c", "rule_based"):
            cache = SENS_CACHE / f"{s['name']}__{c}.json"
            data = json.loads(cache.read_text())
            rev = np.array([r["metrics"]["total_net_revenue_eur"] for r in data.values()])
            row[c] = {"mean": float(rev.mean()), "median": float(np.median(rev))}
        row["gate_minus_rule_mean"] = row["gate_c"]["mean"] - row["rule_based"]["mean"]
        matrix["scenarios"][s["name"]] = row
    OUT_PATH.write_text(json.dumps(matrix, indent=2))
    print(f"wrote {OUT_PATH}")
    for name, row in matrix["scenarios"].items():
        print(
            f"  {name}: gate {row['gate_c']['mean']:,.0f} rule {row['rule_based']['mean']:,.0f} "
            f"diff {row['gate_minus_rule_mean']:+,.0f}"
        )


if __name__ == "__main__":
    main()
