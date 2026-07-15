# Data audit — `iaew-marktdaten.db` (2026-07-15)

Source: the private IAEW database `iaew-marktdaten.db` (SQLite, 5.2 GB, produced by the
IAEW `meerkat` pipeline; EEX/EPEX SFTP + ENTSO-E + Netztransparenz + regelleistung.net;
configured via `$MARKET_DATABASE_PATH` — not distributed with this repository).
All inspection was read-only (`file:...?mode=ro`). The database is **never** modified by
this project.

## 1. Tables selected for the VPP framework

| Series | Table | Native resolution | Timezone of stored naive timestamps | Coverage | Quality notes |
|---|---|---|---|---|---|
| **DAA prices (canonical)** | `day_ahead_prices` (ENTSO-E, DE-LU) | 60 min until 2025-09-30, **15 min from 2025-10-01** (SDAC MTU switch) | **Europe/Berlin local naive** (spring DST day has 23 hourly rows; verified solar-dip at local 13–14) | 2023-01-01 → 2026-07-11, 1287 days, 0 NULLs, 6 duplicate timestamps (DST autumn artifacts) | Use as canonical DAA source. |
| DAA prices (cross-check) | `auction_spot_prices` (EEX wide, hourly + block cols) | 60 min | local product columns (`hour 1..24`) | 2023-01-01 → 2025-09-30, **only 641/1004 days** | Incomplete — cross-check only. |
| **IDA1 prices** | `pan_european_ida1_prices` (wide) | 15 min | local product columns `hour H qQ`, incl. proper `hour 3a/3b` DST columns | 2024-06-14 → 2026-07-13 (696/696 days in study window) | Use wide table. |
| **IDA2 prices** | `pan_european_ida2_prices` (wide) | 15 min | local product columns | 2024-06-14 → …, **13 missing days** in window | Missing days: 2024-07-04, 2024-08-09, 2024-09-21/22, 2024-10-19, 2025-02-04, 2025-05-03, 2025-05-17, 2025-07-16, 2025-09-09, 2025-12-10, 2026-04-16, 2026-04-22. |
| **IDA3 prices** | `pan_european_ida3_prices` (wide) | 15 min | local product columns `hour 13..24 qQ` (delivery 12:00–24:00 local) | 2024-06-14 → …, **10 missing days** | Missing: 2024-06-25, 2024-07-03, 2024-10-27, 2024-10-31, 2025-01-17, 2025-01-22, 2025-09-09, 2025-09-18, 2026-04-01, 2026-04-22. |
| Legacy DE 15:00 IDA | `intraday_auction_spot_prices` (wide) | 15 min | local product columns | 2023-01-01 → 2026-07-13 | Pre-dates pan-EU IDAs; same 15:00 D−1 gate as IDA1. Optional extension of IDA1 history back to 2023. |
| **IDC indices** | `intraday_continuous_indices` | per delivery product: 15/30/60 min; ID1, ID3, IDFULL (+ Base/Peak) | **UTC naive** (verified: series starts 22:00; solar dip at 10–12 UTC) | 2020-07-20 → **2026-05-10**, no NULL prices in study window, no duplicates | ID1 = VWAP of trades in last hour before delivery; ID3 = last 3 h; IDFULL = whole session. `IndexVolume` in MWh. |
| IDC statistics | `intraday_continuous_statistics` | per product 15/30/60 min + blocks | **UTC** (explicit ISO `Z` strings) | 2022-12-31 → 2026-07-12 | low/high/last/VWAP + buy/sell volumes per product — supports spread-model calibration (Baseline B). |
| **Imbalance price (reBAP)** | `netztransparenz_nrv_saldo_reBAP_Qualitaetsgesichert` | 15 min | **UTC** (explicit `Zeitzone=UTC` column) | 2023-01-01 → **2026-06-23** | Quality-assured reBAP; `unterdeckt`=`ueberdeckt` (single price). EUR/MWh. |
| Imbalance price (fallback) | `imbalance_prices` (ENTSO-E) | 15 min | Europe/Berlin local naive (92 rows on spring DST day) | 2023-01-01 → 2026-07-11 | `Long`=`Short` — single-price reBAP mirror; use to extend reBAP beyond 2026-06-23. |
| ID-AEP index | `netztransparenz_id_aep` | 15 min | UTC (explicit) | 2023-01-01 → 2026-07-12 | TEXT values with comma decimals — needs parsing. Alternative IDC execution proxy. |
| **Zone RES forecast (day-ahead)** | `wind_and_solar_forecast` (ENTSO-E) | 15 min | local naive | 2023-01-01 → 2026-07-11 | Solar / Wind Offshore / Wind Onshore, MW, DE-LU zone. |
| Zone RES forecast (intraday) | `intraday_wind_and_solar_forecast` | 15 min | local naive | same | Intraday forecast update. |
| Zone RES actuals | `generation` (ENTSO-E) | 15 min | local naive | same | Per-technology actual generation, MW. |
| Zone forecast errors | `renewable_forecast_errors_15min` | 15 min | local naive | same | Precomputed actual/forecast/error per technology — basis for the synthetic site-level forecast-error model. |

Also present but not used initially: FCR/aFRR/mFRR capacity & energy markets, PICASSO,
frequency, fuel prices, load, cross-border flows, redispatch, market premium. These are
future extensions (ancillary-service co-optimization).

## 2. Timezone findings (verified, not assumed)

Verification methods: DST-day row counts (23/92 rows on 2023-03-26 ⇒ local naive;
96 ⇒ UTC) and the summer midday solar price dip (local 12:00–15:00 ⇔ 10:00–13:00 UTC).

* **ENTSO-E tables** (`day_ahead_prices`, `imbalance_prices`, `generation`,
  `wind_and_solar_forecast`, …): timezone-naive **Europe/Berlin local**, DST-safe
  (spring day short, autumn day has duplicated hour → 6 duplicate timestamps in
  `day_ahead_prices`). The reader must localize with explicit ambiguous/nonexistent
  handling and convert to UTC.
* **EEX/EPEX wide tables**: `Delivery day` (midnight local) + local product columns.
  IDA wide tables carry explicit `hour 3a`/`hour 3b` columns for the 25-h autumn day —
  exact DST treatment is possible from the wide format.
* **EEX-derived `*_ordered` tables**: **artifacts — do not use.**
  * `pan_european_ida3_prices_ordered` maps delivery 12:00–24:00 onto 00:00–11:45
    (12 h shift).
  * `pan_european_ida{1,2}_prices_ordered` write 96 rows even on 92/100-quarter DST
    days (invalid rows).
  * `intraday_auction_spot_prices_ordered` silently drops both DST days each year.
  * → This project reads the **wide** tables and performs its own DST-correct reshape.
* **IDC tables** (`intraday_continuous_indices`, `intraday_continuous_statistics`):
  **UTC naive** / explicit UTC.
* **Netztransparenz tables**: explicit `Zeitzone=UTC` columns.

**Canonical internal convention:** all internal timestamps are UTC (`pandas.Timestamp`,
tz-aware). Delivery products are identified by `(delivery_start_utc, duration)`;
market-local wall time is derived via `Europe/Berlin` only at the calendar/reporting
boundary. No day is ever assumed to have 96 quarter-hours.

## 3. Interpretation of market data

* **DAA**: `day_ahead_prices` = SDAC DE-LU clearing price. Hourly products until
  2025-09-30, 15-min products from 2025-10-01 (both regimes must be supported; the
  product-resolution switch is data-driven, not configured blindly).
* **IDA1/2/3**: EPEX pan-European intraday auctions (go-live 2024-06-13 for delivery
  2024-06-14). Gates (local): IDA1 15:00 D−1 (full day D, 96 QH), IDA2 22:00 D−1
  (full day D), IDA3 10:00 D (delivery 12:00–24:00 D, 48 QH). The database distinguishes
  all three — no approximation needed.
* **IDC**: no order book / trade-by-trade data. Available: ID1, ID3, IDFULL indices per
  delivery product (VWAP over trailing windows) + per-product session statistics
  (high/low/VWAP/last, buy/sell volume). ⇒ **price-taker execution model** (Baseline A):
  a trade decided at lead time ℓ before delivery executes at the corresponding index
  (ID1 for ℓ ≤ 1 h, ID3 for ℓ ≤ 3 h, IDFULL otherwise) plus configurable transaction cost
  and volume cap; optional spread model (Baseline B) calibrated from session statistics.
  This is an approximation and is labelled as such — no order-book realism is claimed.
* **Imbalance**: quality-assured reBAP at 15 min (single price) — the *actual* German
  imbalance price, not an approximation, until 2026-06-23; ENTSO-E mirror to extend.
  Optional stylized asymmetric penalty available as a robustness configuration.

## 4. Usable study window and split (defaults, configurable)

All markets (DAA, IDA1/2/3, IDC indices, reBAP, zone RES) overlap
**2024-06-14 → 2026-05-10** (696 days; limited by IDA go-live and IDC index end).

Default chronological split:

* train 2024-06-14 → 2025-10-31 (≈16.5 months; includes the DAA 15-min switch),
* validation 2025-11-01 → 2026-01-31 (3 months),
* test 2026-02-01 → 2026-05-10 (≈3.3 months).

Missing IDA2/IDA3 auction days (list above) are handled as "auction not held":
the event yields no fills and the position simply carries forward.

## 5. Renewable site data

The database has **zone-level** DE-LU data only — no site-level profiles. Site profiles
come from **Renewables.ninja** (API reachable; `RENEWABLES_NINJA_TOKEN` **not currently
set** — must be provided by the user for real downloads; never committed).

Until the token is provided, a clearly-labelled **synthetic site profile fallback**
scales ENTSO-E zone-level Wind Onshore / Solar actuals to the configured site
capacities (zone profiles are smoother than single-site profiles — documented
limitation, swap-in replacement once real profiles are downloaded).

**Forecasts:** no historical site-level forecasts exist. The framework therefore uses a
configurable synthetic forecast-error model whose error statistics (bias/σ/autocorrelation
per technology and lead time) are fitted on the **training partition** of the ENTSO-E
zone forecast-error table, then applied to site actuals. Perfect foresight and
persistence modes exist for benchmarking/debugging only.

## 6. Data-quality risk register

| Risk | Impact | Mitigation |
|---|---|---|
| `*_ordered` EEX tables corrupt (shift/DST) | wrong prices if used | read wide tables, own reshape, regression tests vs. hand-checked days |
| 6 duplicate timestamps in ENTSO-E tables (autumn DST) | double rows | DST-explicit localization (`ambiguous` resolution by position), dedup, tests |
| IDA2/IDA3 missing days (13/10) | episode gaps | "auction not held" event semantics; validation report |
| DAA resolution switch 2025-10-01 | action-space mismatch | product schema derived from data; 96-slot action mapped to native products |
| IDC index ends 2026-05-10, reBAP 2026-06-23 | window end | study window ends 2026-05-10 |
| No site-level history | synthetic profiles/forecasts | Renewables.ninja + labelled fallback; leakage-safe error model |
| reBAP published ~months later (quality-assured) | in reality unknown at delivery | settlement uses it ex-post only; observations never contain future reBAP |
