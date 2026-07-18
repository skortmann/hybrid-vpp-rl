# Reproducibility

Everything below runs offline against the deterministic synthetic market
database unless a real database is configured. Runnable modules are
configured by editing the marked `CONFIG` block of constants at the top
of each module; there are no command-line flags.

## Environments

```bash
uv sync                        # developer default (dev group): tests, benchmark, RL
uv sync --no-dev               # minimal: synthetic data, rule-based simulation, evaluation
uv sync --no-dev --extra rl    # minimal + training / loading released checkpoints
uv sync --no-dev --extra optimization   # minimal + MILP benchmark (HiGHS)
```

## Regenerating the published tables

The canonical result summary is rebuilt from committed artifacts only —
no model rollouts, no private data:

```bash
uv run python -m hybrid_vpp.evaluation.export_results
# -> results/final_results.json, results/final_results.csv
```

Provenance chain: tag `rl-frontier-v1` freezes the first research phase
(five-seed SAC hybrid, locked test evaluation), tag `robust-rl-final`
freezes the second (selection/ensemble/gate study, pre-registered
confirmation). The committed evidence lives under `artifacts/` (per-day
revenue tables, checkpoint matrix, fold and gate results, the frozen test
evaluation, the one-shot confirmation) and `reports/`.

## Re-running the evaluations

With the real market database configured (`MARKET_DATABASE_PATH`) and the
released checkpoints in place, each analysis is one module run, cached
per (candidate, day):

```bash
uv run python -m hybrid_vpp.evaluation.run_baselines       # baselines over a date range
uv run python -m hybrid_vpp.evaluation.checkpoint_matrix   # 35 checkpoints x 92 validation days
uv run python -m hybrid_vpp.evaluation.ensemble_eval       # ensemble variants + disagreement
uv run python -m hybrid_vpp.evaluation.gate_eval           # safety gates (LOBO-calibrated)
uv run python -m hybrid_vpp.evaluation.milp_variants_eval  # benchmark MILP variants
uv run python -m hybrid_vpp.evaluation.sensitivity_eval    # zero-shot robustness matrix
uv run python -m hybrid_vpp.evaluation.selection_report    # selection analysis report
uv run python -m hybrid_vpp.evaluation.ensemble_report     # ensemble analysis report
uv run python -m hybrid_vpp.evaluation.robust_plots        # figures under docs/assets/robust
```

`hybrid_vpp.evaluation.final_confirmation` re-runs the pre-registered
one-shot test confirmation; it refuses to overwrite an existing result by
design.

## Retraining the promoted policies

```bash
# configs/train_sac_hybrid.yaml is the promoted recipe; seeds 0..4
uv run python -m hybrid_vpp.training.train
```

Training uses SAC on strategic actions (`act-v5`, gain range 1.25,
deterministic rule-based dispatch) for 300k steps per seed under
penalized-imbalance economics. Runs record a config snapshot, seed, host,
git metadata, and data provenance. Expected validation behavior (92
winter days): individual eval-best checkpoints in the 54.3–55.1k EUR/day
mean range, mean-action ensemble above every member (≈55.6k), promoted
gate variant P(outperform rule-based) ≈ 0.9 by block bootstrap. Exact
reproduction of the released checkpoints additionally requires the real
market database and matching library versions (see `uv.lock`).

## Released model assets

The five promoted checkpoints (`sac_hybrid_seed0.zip` …
`sac_hybrid_seed4.zip`, ~4.5 MB each) are attached to the GitHub release.
Place them in `models/` and run:

```bash
uv run python examples/run_deployment_controller.py
```

The repository works without them: the quick start, the baselines, the
test suite, and the MILP benchmark are checkpoint-free.

## Data requirements

| Scenario | Requirement |
|---|---|
| Quick start, tests, CI | none — synthetic database generated on demand (byte-reproducible per seed and config hash) |
| Published market results | the private IAEW market database (`MARKET_DATABASE_PATH`); schema validated against `src/hybrid_vpp/data/schema_manifest.json` |
| Site profiles from measurements | `RENEWABLES_NINJA_TOKEN` (optional; zone-scaled fallback otherwise) |

## Seeds and determinism

Synthetic data: seed 42 (configurable), byte-identical regeneration.
Training: `training.seed` 0–4 for the released policies. Evaluation
rollouts are deterministic (deterministic policies, fixed days); the
bootstrap statistics use fixed RNG seeds. Timestamps are tz-aware UTC
throughout; DST-sensitive behavior is regression-tested.
