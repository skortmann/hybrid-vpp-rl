# Changelog

## v0.2.0 (2026-07-18)

Research-consolidation release: the repository now presents the completed
study and its promoted deployment design.

### Added
* Strategic action mode (`act-v5`): seven economic decision variables
  translated deterministically through the rule-based structure, with an
  interiorized gain range and optional deterministic rule-based dispatch
  (the promoted hybrid architecture).
* SAC/TQC/CrossQ support through a common algorithm adapter; the promoted
  training recipe is published as `configs/train_sac_hybrid.yaml`.
* Policy ensembles in action space, safety gates (disagreement threshold,
  confidence scaling, bounded residual), and the auditable
  `EnsembleDeploymentController` with missing-data and out-of-range
  fallbacks and a replayable decision log.
* Blocked temporal validation, predefined selection rules,
  leave-one-block-out reliability analysis, and a hierarchical
  seed-day bootstrap for paired controller comparisons.
* Benchmark MILP variants (turnover-penalized, forecast-derated) and a
  zero-shot sensitivity suite (grid limits, storage sizing, forecast
  quality, deviation-penalty sweep).
* Canonical machine-readable results (`results/final_results.{json,csv}`)
  regenerated from committed artifacts by
  `hybrid_vpp.evaluation.export_results`; final study report and phase
  analyses under `reports/`.
* Public controller API in `hybrid_vpp.controllers`; runnable examples
  under `examples/`.

### Changed
* Penalized-imbalance economics (historical reBAP plus a 25 EUR/MWh
  deviation penalty) are the default evaluation setting for all
  controllers.
* Dependencies split into a minimal core plus `rl`, `jax`, `tracking`,
  `optimization`, and `gurobi` extras; a minimal install runs the
  synthetic quick start, rule-based simulation, and basic evaluation.

### Removed
* The research-campaign orchestration (run registry, worker, supervisor,
  systemd unit) and the internal research diary; both remain available in
  history at tag `robust-rl-final`. Training exposes a plain JSON
  heartbeat instead.


## v0.1.0 (2026-07-15)

Initial release.

### Physical model
* Wind + PV parks with independent curtailment, oversized relative to the
  grid connection; profiles from Renewables.ninja or a zone-scaled fallback.
* BESS with charge/discharge efficiencies, SoC window, power ratings,
  self-discharge, and throughput-based degradation cost.
* Common point of connection with export/import limits enforced by a
  feasibility projection (exact weighted QP or priority heuristics); every
  correction recorded with requested/applied values and reason; technical
  vs. economic curtailment tracked separately.

### Markets
* German day-ahead auction (hourly products, 15-minute products from the
  SDAC MTU switch), pan-European intraday auctions IDA1/2/3 as independent
  sessions, intraday continuous trading with configurable decision cadence
  and gate-closure lead. DST-exact market calendar (23/24/25-hour days).
* Price-taker execution at historical clearing prices / IDC indices
  (ID1/ID3/IDFULL by lead time) with volume caps and transaction costs.
* Append-only position book per quarter-hour product; per-component cash
  ledger; imbalance settlement at the historical reBAP (single price) or
  stylized alternatives.

### Data
* Read-only adapter for the IAEW market database with UTC normalization,
  DST-correct reshaping of wide auction tables, and Parquet caching.
* Automatic fallback to a deterministic synthetic drop-in database (same
  schema and reader code) with documented statistical structure; provenance
  recorded in all run metadata.
* Renewables.ninja download client with caching and request metadata.

### Reinforcement learning
* Gymnasium environment with event-driven decisions, fixed-size masked
  action tensor, leakage-guarded observations (publication-time guards,
  issue-time-indexed forecasts), chronological train/validation/test splits.
* Stable-Baselines3 PPO training with checkpointing, validation-based model
  selection, and Weights & Biases / TensorBoard tracking.

### Benchmarks
* Do-nothing and rule-based baselines; rolling-horizon MILP benchmark
  (PyOptInterface; Gurobi with HiGHS fallback) using the same forecasts as
  the RL agent; perfect-foresight MILP upper bound.

### Known limitations
* IDC execution is an index-price price-taker approximation (no order book,
  no partial fills, no market impact).
* One intraday zone-forecast snapshot per interval; intraday forecast error
  does not shrink with lead time.
* Synthetic market data are for development and testing — not a basis for
  revenue claims about the German market.
