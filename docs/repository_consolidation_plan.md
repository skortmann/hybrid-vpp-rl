# Repository consolidation plan

Audit date: 2026-07-18. Repository: `github.com/skortmann/hybrid-vpp-rl`
(default branch `main`, unprotected, no open pull requests, release
`v0.1.0` published). Goal: consolidate the research branches into one
authoritative default branch that presents completed research software,
while preserving scientific results, reproducibility, and full history.

## Branch and tag inventory (pre-consolidation)

| Ref | Head | Relationship | Decision |
|---|---|---|---|
| `main` | `79ec6dc` (21 commits) | ancestor of both research branches; nothing unique | KEEP (default; merge target) |
| `research/advanced-rl` | `17b002e` | ancestor of `research/robust-rl-selection` | MERGE (already contained); DELETE after merge |
| `research/robust-rl-selection` | `6cbe81d` (53 commits, superset of everything) | consolidation source | MERGE via `release/publication-cleanup`; DELETE after merge |
| tag `v0.1.0` | initial release | — | KEEP |
| tag `rl-frontier-v1` | `17b002e`, first terminal research state | — | KEEP |
| tag `robust-rl-final` | `6cbe81d`, second terminal research state | to create | CREATE + KEEP (immutable pre-cleanup state) |
| branch `archive/pre-publication-consolidation` | `6cbe81d` | backup | CREATE + KEEP (never force-pushed) |

Because the three branches are strictly linear (verified with
`git merge-base --is-ancestor`), no cherry-picking is required: every
validated commit is preserved by branching `release/publication-cleanup`
from `6cbe81d` and merging that into `main`.

## Content classification

Tracked content is small (≈1.9 MB, 143 files). The audit found no
committed databases, checkpoints, experiment-tracking caches, credentials,
machine-specific paths, or AI-assistant material in tracked files.

### KEEP (required for use or reproducibility; public-safe)

* `src/hybrid_vpp/` core: `assets/`, `markets/`, `sim/`, `data/`,
  `forecasts/`, `controllers/`, `envs/`, `config/`, `core/` — the
  validated implementation.
* `src/hybrid_vpp/evaluation/` — all modules; they regenerate every
  published number (baselines, checkpoint matrix, ensembles, gates,
  sensitivity, final confirmation, plots, reports).
* `src/hybrid_vpp/training/`: `train.py`, `algorithms.py`, `evaluate.py`,
  `datasets.py`, `experiments.py` (lightweight run/provenance runner).
* All five action schemas (`act-v1`…`act-v5`): schema-versioned,
  test-pinned, and required to interpret the study and load the released
  checkpoints. `act-v1` is documented as a known-broken reference, which
  is a scientific statement, not dead code.
* `artifacts/`: compact result provenance — `test_evaluation.json`
  (frozen phase-1 test evidence, consumed by the final-confirmation
  code), `anchors_v2.json`, `robust_selection/*.{json,csv}`,
  `failure_days/*`. Required to verify published numbers without
  re-running training.
* `reports/`: the two terminal-state reports (`final_rl_results.*`,
  `final_robust_rl_results.*`), the five phase analyses (checkpoint
  selection, ensembles, failure days, regimes, MILP variants), and the
  study report. Distinct analyses, not duplicates.
* `experiments/baseline_v1.json`, `experiments/diagnostics_baseline_ppo.json`
  — compact provenance of the recorded starting point.
* `tests/` except the supervisor tests (below); `configs/*.yaml`;
  `.github/` (CI is already minimal: format, lint, offline tests, build,
  strict docs); `LICENSE`, `CITATION.cff`, `CONTRIBUTING.md`,
  `SECURITY.md`, `CODE_OF_CONDUCT.md`, `CHANGELOG.md`.

### REMOVE from the release branch (preserved at tag `robust-rl-final`)

| Item | Reason |
|---|---|
| `src/hybrid_vpp/training/research_state.py`, `worker.py`, `supervisor.py` | autonomous research-campaign orchestration (registry state machine, heartbeats, watchdog); only useful for the completed campaign; not needed for training, evaluation, deployment, or reproduction. `train.py`'s optional heartbeat is inlined to drop the dependency. |
| `tests/unit/test_research_supervisor.py` | tests only the removed infrastructure |
| `docs/research_supervision.md` | documents the removed infrastructure |
| `deploy/hybrid-vpp-rl-supervisor.service` (and `deploy/`) | systemd unit for the removed supervisor |
| `docs/robust_rl_research_plan.md`, `docs/robust_rl_research_log.md` | internal research diary; superseded by `docs/results.md` + `reports/final_study_report.md`; full text remains readable at the tag |
| `artifacts/screening_table.{csv,md}` | intermediate screening under the superseded env-v1 economics; conclusions live in the final reports |

### REWRITE

* `README.md` — final-software framing: capabilities, install (core +
  extras), synthetic quick start, real-data configuration, training,
  evaluation, deployment controller, headline results, limitations.
* `docs/index.md`, `mkdocs.yml` nav — final documentation structure.
* `docs/rl_research.md` → condensed research-summary page (stale status
  language removed; mechanism findings kept).
* `reports/complete_study_summary.md` → `reports/final_study_report.md`
  (canonical study report; content unchanged).
* `pyproject.toml` — version 0.2.0; dependency split: minimal core
  (synthetic data + rule-based simulation + basic evaluation) with
  optional extras `rl`, `jax`, `tracking`, `optimization`, `gurobi`.

### CREATE

* `docs/results.md`, `docs/limitations.md`, `docs/reproducibility.md`,
  `docs/controllers.md`.
* `results/final_results.json` + `results/final_results.csv` — one
  canonical machine-readable summary (generated by
  `hybrid_vpp.evaluation.export_results` from the artifacts).
* `examples/quickstart.py`, `examples/evaluate_baselines.py`,
  `examples/run_deployment_controller.py` (runnable as-is; tunables as
  module constants, consistent with the project's no-CLI-flags
  convention).
* `configs/train_sac_hybrid.yaml` — the promoted training recipe.
* Public controller API: `DoNothingController`, `RuleBasedController`,
  `OptimizationController` (benchmark), plus `PolicyEnsemble`,
  `SafetyGate`, and `EnsembleDeploymentController` (deployment) exported
  from `hybrid_vpp.controllers`.
* Release `v0.2.0` with the five promoted SAC checkpoints (4.5 MB each)
  attached as assets, so the ensemble controller is reproducible without
  retraining.

### Decisions on open design questions

* **Action schema** stays seven-dimensional (`act-v5`): the released
  checkpoints require it, and dispatch dimensions 4–6 are inert under the
  promoted fixed-dispatch configuration. This is documented once, in
  `docs/controllers.md`; no new four-dimensional schema is introduced
  without models trained on it.
* **Command surface** remains the runnable-module convention
  (`python -m hybrid_vpp.<module>` with a marked CONFIG block per
  module); the project deliberately has no flag-parsing CLI.
* **CI** is kept as-is: it already runs offline (synthetic fallback), and
  requires no private data, tokens, W&B, or Gurobi.

## Integration order

1. Create and push tag `robust-rl-final` and branch
   `archive/pre-publication-consolidation` at `6cbe81d`.
2. Branch `release/publication-cleanup` from `6cbe81d`.
3. Apply the removals, rewrites, and additions above in reviewable
   commits; run the full test suite, lint, strict docs build, and package
   build after each group.
4. Validate the clean-user workflow in a fresh clone and environment.
5. Push, open a pull request into `main`, merge after CI.
6. Delete the fully-merged `research/*` branches; keep the archive branch
   and all tags; publish release `v0.2.0`.
