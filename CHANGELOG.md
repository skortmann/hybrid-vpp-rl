# Changelog

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
