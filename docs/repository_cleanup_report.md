# Repository cleanup report

Consolidation of the research branches into one publishable default
branch, executed 2026-07-18 on `release/publication-cleanup` per
[repository_consolidation_plan.md](repository_consolidation_plan.md).

## Branch inventory

| Before | After |
|---|---|
| `main` (21 commits) | `main` — default, receives the consolidation merge |
| `research/advanced-rl` (`17b002e`) | deleted after merge; head preserved by tag `rl-frontier-v1` and by `main` history |
| `research/robust-rl-selection` (`6cbe81d`, 53 commits, strict superset) | deleted after merge; head preserved by tag `robust-rl-final` and branch `archive/pre-publication-consolidation` |
| — | `archive/pre-publication-consolidation` @ `6cbe81d` (backup, kept) |
| tags `v0.1.0`, `rl-frontier-v1` | kept; `robust-rl-final` and `v0.2.0` added |

All three pre-existing branches were strictly linear
(`main` ⊂ `research/advanced-rl` ⊂ `research/robust-rl-selection`), so the
consolidation branch was created from the superset head and **no commit
was cherry-picked, squashed, or lost**. Scientific history remains fully
readable in `git log` and at the phase tags.

## Consolidation commits

```
379e723 refactor: remove research-campaign orchestration from the package
dac35ca feat: add canonical machine-readable result summary
b5e3969 feat: add public controller API, examples, and training recipe
7d9db23 build: split optional dependencies and bump version to 0.2.0
b1fc621 docs: rewrite documentation for the consolidated release
ee60989 style: apply formatting to result exporter
642ea36 docs: correct minimal-install commands
```

## Removed (preserved in history at `robust-rl-final`)

* `src/hybrid_vpp/training/research_state.py`, `worker.py`,
  `supervisor.py` — the autonomous research-campaign orchestration (run
  registry, heartbeats, watchdog, retry state machine). Not required for
  training, evaluation, deployment, or reproduction. `train.py` now
  writes a plain JSON heartbeat with no orchestration dependency.
* `tests/unit/test_research_supervisor.py` (16 tests of the removed
  infrastructure) and `docs/research_supervision.md`.
* `deploy/hybrid-vpp-rl-supervisor.service` (systemd unit for the removed
  supervisor).
* `docs/robust_rl_research_plan.md`, `docs/robust_rl_research_log.md` —
  the internal research diary, superseded by `docs/results.md`,
  `docs/rl_research.md`, and `reports/final_study_report.md`.
* `artifacts/screening_table.{csv,md}` — intermediate screening under the
  superseded pre-penalty economics.

The audit found nothing else to purge: no committed databases, model
checkpoints, experiment-tracker caches, credentials, machine-specific
paths, or assistant-tool files existed in tracked content.

## Added

* `results/final_results.json` / `.csv` — canonical machine-readable
  summary (22 rows: controller x split with means, medians, paired
  bootstrap CIs, info gaps, provenance tag), regenerated from committed
  artifacts by `hybrid_vpp.evaluation.export_results`.
* Public controller API in `hybrid_vpp.controllers`
  (`DoNothingController`, `RuleBasedController`, `OptimizationController`
  lazily, `PolicyEnsemble`, `SafetyGate`, `EnsembleDeploymentController`).
* `examples/quickstart.py` (offline synthetic day) and
  `examples/run_deployment_controller.py` (promoted controller from
  released checkpoints).
* `configs/train_sac_hybrid.yaml` — the promoted training recipe.
* `docs/controllers.md`, `docs/results.md`, `docs/limitations.md`,
  `docs/reproducibility.md`; `docs/rl_research.md` rewritten as a final
  research summary; README updated for the completed project.
* `reports/complete_study_summary.md` renamed to
  `reports/final_study_report.md` (canonical study report).

## Configuration and dependency changes

* Version `0.1.0` → `0.2.0` (pre-1.0 kept deliberately: the API is
  stable but young). Changelog entry added.
* Dependencies split: minimal core (numpy, pandas, pydantic, pyyaml,
  gymnasium, matplotlib, requests, pyarrow) plus extras `rl`
  (SB3/torch/contrib/tensorboard), `jax` (sbx), `tracking` (wandb/weave),
  `optimization` (PyOptInterface+HiGHS), `gurobi`. The `dev` group is the
  batteries-included developer environment used by CI (unchanged CI
  workflow). A minimal install (`uv sync --no-dev`) contains no ML or
  solver packages and runs the synthetic quick start.
* Configs: the four scenario configs kept; `train_sac_hybrid.yaml` added.
  No stale or duplicate configs were found.
* Action schema decision: `act-v5` stays seven-dimensional for released-
  checkpoint compatibility; dispatch dimensions 4–6 are documented as
  inert under the promoted fixed-dispatch configuration
  (`docs/controllers.md`). No adapter is needed — deployment and legacy
  models share one schema.

## Public API migration notes

* `from hybrid_vpp.envs.deployment import DeploymentController` →
  preferred: `from hybrid_vpp.controllers import
  EnsembleDeploymentController` (same class).
* `hybrid_vpp.training.research_state` / `worker` / `supervisor` no
  longer exist; running experiment batches is
  `hybrid_vpp.training.experiments`; process supervision is out of scope
  (the training heartbeat file remains machine-readable).
* Everything else is unchanged; no behavior of the simulator,
  environments, controllers, or evaluation modules was modified.

## Validation

* Tests: 173 collected (was 189; the 16 removed tests covered only the
  removed orchestration); offline run **149 passed, 24 skipped
  (real-database tests), 0 failed** in the consolidation clone and in a
  fresh clean-user clone.
* Lint and format: clean. Package build: sdist + wheel `0.2.0` build
  successfully. Docs: `mkdocs build --strict` clean.
* Clean-user workflow (fresh clone, no environment variables): minimal
  install → synthetic quick start ✓; dev install → full test suite ✓,
  package build ✓, strict docs ✓, canonical results regenerate ✓;
  deployment example with the five release checkpoints staged in
  `models/` runs end-to-end (132 logged decisions, 0 fallbacks).
* Secret/path scan on the final tree: no credentials, tokens, private
  data, or machine-specific paths.
* Repository size: 1.84 MB tracked in 162 files (before: 1.87 MB in
  161 files) — the repository was already artifact-hygienic; this
  cleanup was about structure, framing, and the dependency surface.

## Known remaining limitations

* The published headline numbers derive from the private market database;
  a clean-room user can regenerate the summary tables and validation
  logic, run everything on synthetic data, and load the released
  checkpoints, but cannot re-derive the historical revenue figures
  without database access (documented in `docs/reproducibility.md`).
* `docs/data_audit.md` retains its dated audit format on purpose — it is
  the authoritative record of source-data quirks.
* Scientific limitations of the study itself are in
  `docs/limitations.md`.
