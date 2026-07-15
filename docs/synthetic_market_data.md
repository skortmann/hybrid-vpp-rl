# Synthetic market database

## Why it exists

The framework is developed against the private IAEW database
(`iaew-marktdaten.db`). External researchers, CI, and quick-start users do
not have that file. The synthetic market database is a **drop-in
replacement**: same tables, same column names, same timestamp conventions
(the committed contract is `src/hybrid_vpp/data/schema_manifest.json`), so
every controller, environment, and training entry point runs unchanged.

Synthetic data are for testing, software demonstrations, algorithm
development, reproducibility examples, and sensitivity studies. **They must
not be used to make claims about actual German market revenues** without
validation against real historical data. Every generated database carries a
`synthetic_metadata` table (`synthetic: true`, seed, config hash, schema
and generator versions, creation time, git commit) — identification never
relies on the filename.

## Source selection (resolver)

```yaml
data:
  market_database:
    mode: auto          # real | synthetic | auto
    path: ${MARKET_DATABASE_PATH}     # null -> env var -> known default
    synthetic_path: data/generated/synthetic-iaew-marktdaten.db
    create_if_missing: true
    fallback_on:
      missing_file: true
      invalid_schema: true
      insufficient_coverage: true
```

* `real` — require the real database; fail with an actionable error
  otherwise; never silently switch.
* `synthetic` — always use (and create if missing) the synthetic database.
* `auto` — use the real database when it exists, is readable, passes the
  schema-manifest validation, and (when a required period is supplied)
  covers it; otherwise generate/reuse the synthetic one. The active source
  is logged at INFO/WARNING and recorded in run metadata
  (`runs/<run>/metadata.json`, wandb config, evaluation metadata JSON) with
  provenance `real` / `synthetic` / `synthetic_calibrated`.

Python API: `hybrid_vpp.data.resolver.resolve_market_database(...)`;
runnable modules: `hybrid_vpp.data.schema_manifest` (inspect/regenerate the
manifest), `hybrid_vpp.data.synthetic_market` (generate),
`hybrid_vpp.data.resolver` (show which source would be used),
`hybrid_vpp.data.inspect_database` (coverage report).

## What is generated

All series read by `MarketDataStore`, in the exact real-schema tables:

| Table | Content |
|---|---|
| `day_ahead_prices` | DAA prices, hourly before / 15-min after the configured SDAC switch date; naive local timestamps incl. duplicated autumn hour |
| `pan_european_ida{1,2,3}_prices` | wide local product columns (`hour H qQ`, `hour 3a/3b`); IDA3 covers hours 13–24 |
| `intraday_continuous_indices` | ID1 / ID3 / IDFULL per 15-min product (UTC) + volume proxy |
| `intraday_continuous_statistics` | bid/ask proxy: low/high = mid ∓ spread/2, VWAP, last, buy/sell volumes |
| `netztransparenz_nrv_saldo_reBAP_Qualitaetsgesichert` | reBAP (single price; optional asymmetric unterdeckt/ueberdeckt) |
| `imbalance_prices` | ENTSO-E-style mirror (local naive) |
| `generation`, `wind_and_solar_forecast`, `intraday_wind_and_solar_forecast` | zone actuals + day-ahead/intraday forecasts (shared error process) |

## Statistical model (lightweight, documented, seeded)

1. **Fundamentals**: load (annual/weekly/intraday shapes + AR(1)), wind CF
   (persistent AR(1), winter-heavy), PV CF (clear-sky × AR(1) clouds);
   residual load = load − wind − PV.
2. **Volatility clustering**: 2-state Markov regime scaling all innovations.
3. **DAA** = convex merit-order function of residual load + AR(1) residual +
   seeded positive/negative spikes (bounded by configured limits); negative
   prices arise endogenously at low residual load.
4. **Forecast errors**: AR errors per technology; intraday errors are
   strictly smaller than day-ahead errors. The same errors populate the zone
   forecast tables and drive intraday price updates — renewable surprises
   move IDA/IDC prices with the documented sign (over-forecast wind ⇒ lower
   intraday prices).
5. **IDA chain**: `ida1 = daa + update₁ + ε₁`, `ida2 = ida1 + update₂ + ε₂`,
   `ida3 = ida2 + update₃ + ε₃` with decreasing update variance (information
   arrival; assumption documented here).
6. **IDC**: mid = ida3 + noise; ID1/ID3/IDFULL are noisy views (ID1 tracks
   the imbalance direction most, being closest to delivery). Spread proxy
   `s_t = max(s_min, f(volatility, |forecast error|))` ⇒ bid = mid − s/2,
   ask = mid + s/2, never negative.
7. **reBAP** = IDC mid + heavy-tailed term aligned with the system-imbalance
   direction (system short when renewables under-deliver ⇒ higher reBAP).
   This is a stylized single-price settlement — it does **not** reproduce
   the full German AEP/reBAP calculation.

Verified properties (see `tests/unit/test_synthetic_market.py`): serial
correlation, intraday and weekday seasonality, negative prices, bounded
spikes, DAA↔IDA↔IDC correlations > 0.6, non-negative spreads, intraday
forecast error < day-ahead error, correct 92/96/100-quarter-hour DST days,
leap days, deterministic regeneration per seed, cache reuse keyed on the
configuration hash and schema version.

## Calibration (optional)

`synthetic_market.calibration.enabled: true` (with a real source database)
may derive aggregate parameters — hourly/monthly price levels, variance,
negative-price frequency, spread distributions — without copying raw
records; the resulting provenance is `synthetic_calibrated`. The current
implementation ships sensible defaults; the calibration hook is the place
to add fitted parameters. Raw proprietary records are never written into
the synthetic database or the repository.

## Reproducibility & speed

Same configuration + seed ⇒ byte-identical table contents (creation
timestamp lives only in metadata). Different seeds ⇒ different
realizations. One year at 15 minutes generates in ≈ 5 s (28 MB); a
two-week test range in < 1 s. Existing databases are reused unless the
configuration hash or schema version changed. Parquet caches are
namespaced per database identity, so real and synthetic caches never mix.

## Quick start without the real database

```bash
uv sync --group dev
uv run python -m hybrid_vpp.data.synthetic_market      # generate (config block)
uv run pytest tests/unit tests/integration/test_synthetic_pipeline.py
uv run python -m hybrid_vpp.evaluation.run_baselines   # point CONFIG_PATH at configs/synthetic_market.yaml
```

With access to the real database:

```bash
export MARKET_DATABASE_PATH=/path/to/iaew-marktdaten.db   # mode: auto picks it up
```
