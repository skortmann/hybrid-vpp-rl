# hybrid-vpp-rl

[![CI](https://github.com/skortmann/hybrid-vpp-rl/actions/workflows/ci.yml/badge.svg)](https://github.com/skortmann/hybrid-vpp-rl/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Reinforcement-learning energy-management and trading framework for a
co-located **wind + PV + battery** hybrid virtual power plant participating
in the German **day-ahead auction (DAA)**, the pan-European **intraday
auctions (IDA1/2/3)**, and **intraday continuous (IDC)** trading.

## Motivation

Hybrid parks are increasingly built **oversized** relative to their grid
connection: installed wind + PV capacity exceeds the export limit, and a
battery decides whether excess generation is stored, shifted, or curtailed.
Operating such a portfolio is a joint market-timing and physical-dispatch
problem across sequential markets with hard gate closures. This framework
provides a technically careful simulation of that problem — commercial
positions and physical dispatch strictly separated, no look-ahead leakage,
every constraint intervention audited — plus baselines and a Gymnasium
environment for reinforcement-learning research on top of it.

## System architecture

```
                    ┌────────────── markets ──────────────┐
   DAA (12:00 D-1) → IDA1 (15:00 D-1) → IDA2 (22:00 D-1) → IDA3 (10:00 D) → IDC (rolling)
                    └───────── append-only position book ─────────┘
                                        │ contracted position
 wind park ──┐                          ▼
 PV park  ───┼─ feasibility projection → grid connection (export/import limits)
 BESS ───────┘   (QP / priority modes)  │ delivered energy
                                        ▼
                       imbalance settlement (reBAP) + cash ledger
                                        ▲
                     RL controller / rule-based / MILP benchmarks
```

* One **event-driven simulator**: auction gates, IDC decisions, physical
  dispatch, and settlement as ordered, DST-exact calendar events.
* **Commercial layer**: additive trades per quarter-hour product, frozen
  after gate closure; per-component cash ledger (each euro booked once).
* **Physical layer**: SoC-consistent battery, independent wind/PV
  curtailment, and a feasibility projection that resolves grid congestion
  (charge / curtail / reduce discharge) without ever violating the power
  balance — every correction is recorded with a reason.

## Key capabilities

* Hourly **and** 15-minute DAA products (the SDAC MTU switch is handled
  from data, not assumed).
* IDA1/2/3 as independent sessions with their own gates and eligibility.
* Configurable IDC decision cadence, opening times, and gate-closure lead.
* Historical German imbalance settlement (reBAP) or stylized alternatives.
* Oversizing analytics: technical vs. economic curtailment, congestion
  charging, grid-utilization and duration-curve metrics.
* Baselines: do-nothing, rule-based, rolling-horizon MILP (PyOptInterface;
  Gurobi or HiGHS), perfect-foresight upper bound.
* SB3 PPO training with validation-based model selection and W&B tracking.
* **Synthetic market database**: a deterministic, clearly-labelled drop-in
  replacement for the private market database — the full test suite,
  examples, and short trainings run offline
  (see `docs/synthetic_market_data.md`).

## Modeling assumptions (current)

* **Price-taker execution** in all markets; auction fills at historical
  clearing prices, IDC fills at ID1/ID3/IDFULL indices by remaining lead
  time plus transaction costs. This framework does **not** reproduce real
  IDC order-book trading — no order book, no partial fills, no market
  impact (see `docs/model.md` for the complete list).
* Forecasts: synthetic site forecasts built by transferring concurrent
  zone-level forecast errors onto site profiles; realized prices become
  observable only after their historical publication times.
* Single centralized portfolio agent; one common point of connection.

## Installation

Requires Python ≥ 3.12 and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/skortmann/hybrid-vpp-rl
cd hybrid-vpp-rl
uv sync --group dev
```

## Quick start (no private data required)

```bash
# 1. generate the synthetic market database (~5 s, deterministic, seed 42)
uv run python -m hybrid_vpp.data.synthetic_market

# 2. run the offline test suite
uv run pytest

# 3. one scripted episode with full accounting output
uv run python -m hybrid_vpp.sim.demo_episode        # set CONFIG_PATH to configs/synthetic_market.yaml

# 4. baseline comparison and a short RL training
uv run python -m hybrid_vpp.evaluation.run_baselines
uv run python -m hybrid_vpp.training.train          # configs/synthetic_market.yaml = short demo run
```

Runnable modules are configured by editing the marked `CONFIG` block of
constants at the top of each module; all domain parameters live in
`configs/*.yaml` (pydantic-validated).

## Environment variables

Copy `.env.example` to `local.env` (gitignored, auto-loaded):

| Variable | Purpose |
|---|---|
| `MARKET_DATABASE_PATH` | path to the real market database (optional — synthetic fallback otherwise) |
| `RENEWABLES_NINJA_TOKEN` | Renewables.ninja API token for site-profile downloads (optional) |
| `WANDB_API_KEY` | Weights & Biases tracking (optional; `training.tracker: none` disables) |

**Market data and API credentials are not distributed with this
repository.**

## Market-data configuration

```yaml
data:
  market_database:
    mode: auto          # real | synthetic | auto
    path: ${MARKET_DATABASE_PATH}
    synthetic_path: data/generated/synthetic-iaew-marktdaten.db
    create_if_missing: true
```

`auto` uses the real database when it exists and passes schema/coverage
validation, otherwise generates the synthetic drop-in. The active source
(`real` / `synthetic`) is logged and recorded in every run's metadata.
Interface documentation: `docs/data_audit.md` (real schema audit) and
`docs/synthetic_market_data.md` (generator, resolver, provenance).

## Renewable site profiles

```bash
# set RENEWABLES_NINJA_TOKEN in local.env, then:
uv run python -m hybrid_vpp.data.renewables_ninja
```

Downloads hourly wind/PV profiles for the configured site (coordinates,
capacities, turbine model, PV geometry), caches raw responses with request
metadata, and builds a 15-minute parquet. Without a token, the framework
uses a clearly-labelled zone-scaled fallback profile so everything still
runs.

## Training, evaluation, tests

```bash
uv run python -m hybrid_vpp.training.train       # PPO; seeds, checkpoints, val-based selection
uv run python -m hybrid_vpp.training.evaluate    # RL vs. all baselines on val/test days
uv run pytest                                    # offline: 107 tests; +22 with the real DB
```

### Example output (validation split, real data, 92 days)

| Controller | Net revenue | Abs. deviation | Full cycles |
|---|---|---|---|
| Do-nothing | €4.96M | 4,520 MWh | 0 |
| Rule-based | €5.19M | 3,460 MWh | 78 |
| MILP (same forecasts as RL) | €5.13M | 2,009 MWh | 161 |
| MILP perfect foresight (upper bound) | €5.50M | 946 MWh | 153 |

Metrics per run include per-market revenues, imbalance/transaction/
degradation costs, curtailment split into technical vs. economic,
congestion-absorbed energy, equivalent full cycles, trading volumes per
market, and action-correction counts.

## Repository structure

```
src/hybrid_vpp/
  config/       pydantic configuration models
  core/         canonical time grid, delivery products, MW/MWh conversions
  data/         market-database adapter, resolver, synthetic generator,
                schema manifest, Renewables.ninja client, site profiles
  markets/      calendar (gates/eligibility), execution, positions, ledger,
                imbalance settlement
  assets/       battery, feasibility projection (grid congestion)
  forecasts/    renewable + price forecast providers (issue-time indexed)
  sim/          deterministic event-driven simulator, scripted demo
  envs/         Gymnasium environment, action layout, observation builder
  controllers/  rule-based, MILP benchmark, do-nothing
  training/     SB3 training + evaluation entry points
  evaluation/   metrics, plots, baseline runner
tests/          unit / integration / leakage suites (offline by default)
configs/        example configurations (values are illustrative)
docs/           MkDocs documentation
```

## Reproducibility

* Chronological train/validation/test splits; episodes never cross splits;
  leakage tests pin publication-time guards.
* Every training run stores a config snapshot, seed, host, git metadata,
  and the market-data provenance (`real`/`synthetic`) under `runs/`.
* The synthetic database regenerates byte-identically from
  (configuration, seed); its metadata table records generator version,
  schema version, config hash, and creation info.
* Evaluation writes per-day CSVs plus a metadata JSON naming the exact
  days, model checkpoint, and data source.

## Known limitations

* IDC index-price execution model (documented approximation).
* Zone-scaled fallback profiles understate single-site variability.
* One intraday zone-forecast snapshot per interval (no issue-time history).
* Static throughput-based battery degradation cost (interfaces ready for
  cycle-based models).

## License

MIT — see [LICENSE](LICENSE).

## Citation

See [CITATION.cff](CITATION.cff). If you use this software in research,
please cite it via the repository metadata.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, quality gates, and
extension points (new markets, assets, controllers, dataset adapters), and
[SECURITY.md](SECURITY.md) for credential handling and vulnerability
reports.
