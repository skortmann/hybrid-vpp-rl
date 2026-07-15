"""Deterministic synthetic market database — drop-in for ``iaew-marktdaten.db``.

Generates a SQLite database with the **exact tables, column names, and
timestamp conventions** of the real IAEW database subset the framework reads
(see ``schema_manifest.json``), so `MarketDataStore` works on it unchanged.
The data are clearly labelled synthetic (``synthetic_metadata`` table) and
must never be presented as actual German market observations.

Statistical structure (lightweight, fully seeded, documented):

* **Fundamentals**: zone load (annual/weekly/intraday shape + AR(1)),
  wind capacity factor (persistent AR(1) through a logistic link, winter-
  heavy), PV capacity factor (clear-sky shape x AR(1) cloud factor);
  residual load = load - wind - PV.
* **Volatility regime**: 2-state Markov chain (calm/volatile) scales all
  innovations — volatility clustering.
* **DAA price** = merit-order function of residual load (convex) + AR(1)
  residual + seeded positive/negative spikes; negative prices arise at low
  residual load (windy nights, sunny weekends). Hourly products before the
  configured 15-minute switch date, quarter-hourly after.
* **Forecast errors**: smooth AR errors per technology for the day-ahead
  and (smaller) intraday zone forecasts; the *same* errors drive the zone
  forecast tables and the intraday price chain, so renewable surprises move
  intraday prices (documented sign: over-forecast wind -> lower IDA/IDC).
* **IDA chain**: ida1 = DAA + error-update + noise; ida2/ida3 add further
  updates with decreasing variance (information arrives, forecast error
  shrinks toward delivery).
* **IDC**: mid = ida3 + noise; ID1/ID3/IDFULL are noisy views of the mid
  with ID1 (closest to delivery) tracking the reBAP direction most;
  statistics table carries a volatility-dependent bid/ask proxy
  (low/high = mid -/+ spread/2), spread >= configured minimum.
* **reBAP** = IDC mid + heavy-tailed term aligned with the system-imbalance
  direction (negative total renewable forecast error -> system short ->
  higher reBAP). Written to the Netztransparenz table (single price, or
  asymmetric unterdeckt/ueberdeckt if configured) and the ENTSO-E mirror.

Generation is a single vectorized pass (~2 s/year) with bulk inserts and
indexes created afterwards. An existing database is reused when its
metadata matches the configuration hash and schema version.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from hybrid_vpp.config.models import MarketsConfig, SyntheticMarketConfig
from hybrid_vpp.core.timegrid import MARKET_TZ, local_day_bounds_utc
from hybrid_vpp.data.schema_manifest import SCHEMA_VERSION
from hybrid_vpp.data.sqlite_market_data import MarketDataStore

log = logging.getLogger(__name__)

GENERATOR_VERSION = "1.0"


def config_hash(cfg: SyntheticMarketConfig) -> str:
    payload = cfg.model_dump_json()
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _git_commit() -> str | None:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            ).stdout.strip()
            or None
        )
    except OSError:
        return None


# --------------------------------------------------------------------- helpers


def _ar1(rng: np.random.Generator, n: int, phi: float, sigma: float) -> np.ndarray:
    eps = rng.normal(0.0, sigma, n)
    out = np.empty(n)
    out[0] = eps[0] / np.sqrt(max(1e-9, 1 - phi * phi))
    for i in range(1, n):
        out[i] = phi * out[i - 1] + eps[i]
    return out


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _spikes(rng, n, rate_per_year, intervals_per_year, lo, hi) -> np.ndarray:
    """Seeded spike process: Poisson starts, geometric duration, lognormal size."""
    out = np.zeros(n)
    n_events = rng.poisson(rate_per_year * n / intervals_per_year)
    starts = rng.integers(0, n, size=n_events)
    for s in starts:
        duration = min(1 + rng.geometric(0.4), 12)
        size = rng.lognormal(mean=np.log(120), sigma=0.6)
        out[s : s + duration] += np.clip(size, lo, hi)
    return out


# ------------------------------------------------------------------ generator


class SyntheticSeries:
    """All generated series on the canonical 15-min UTC grid."""

    def __init__(self, cfg: SyntheticMarketConfig, markets: MarketsConfig) -> None:
        self.cfg = cfg
        start_utc = local_day_bounds_utc(cfg.start)[0]
        end_utc = local_day_bounds_utc(cfg.end - pd.Timedelta(days=1))[1]
        self.grid = pd.date_range(start_utc, end_utc, freq="15min", inclusive="left")
        self.markets = markets
        self._generate()

    def _generate(self) -> None:
        cfg, grid = self.cfg, self.grid
        rng = np.random.default_rng(cfg.random_seed)
        n = len(grid)
        per_year = 365 * 96
        local = grid.tz_convert(MARKET_TZ)
        hour = local.hour.to_numpy() + local.minute.to_numpy() / 60.0
        dow = local.dayofweek.to_numpy()
        doy = local.dayofyear.to_numpy()
        weekend = (dow >= 5).astype(float)

        # ---- volatility regime (2-state Markov: calm / volatile)
        flips = rng.random(n)
        regime = np.empty(n)
        state = 0.0
        for i in range(n):
            if state == 0.0 and flips[i] < 0.002:
                state = 1.0
            elif state == 1.0 and flips[i] < 0.01:
                state = 0.0
            regime[i] = state
        vol = 1.0 + 2.0 * regime

        # ---- fundamentals
        annual = np.cos(2 * np.pi * (doy - 15) / 365.25)  # winter-peaking
        intraday_shape = (
            0.82
            + 0.16 * np.exp(-((hour - 8.5) ** 2) / 8.0)
            + 0.20 * np.exp(-((hour - 19.0) ** 2) / 6.0)
            - 0.10 * np.exp(-((hour - 3.0) ** 2) / 10.0)
        )
        load = cfg.zone_load_mw * (1.0 + 0.10 * annual) * intraday_shape * (
            1.0 - 0.12 * weekend
        ) + _ar1(rng, n, 0.98, 350.0)
        wind_cf = _sigmoid(_ar1(rng, n, 0.9985, 0.055) + 0.5 * annual - 0.6)
        clearsky = np.maximum(0.0, np.sin(np.pi * (hour - 6.0) / 13.0)) ** 1.4
        season_pv = 0.55 - 0.45 * annual  # summer-peaking
        pv_cf = clearsky * season_pv * _sigmoid(_ar1(rng, n, 0.995, 0.10) + 1.0)
        wind_mw = wind_cf * cfg.zone_wind_capacity_mw
        pv_mw = pv_cf * cfg.zone_pv_capacity_mw
        residual = load - wind_mw - pv_mw
        residual_norm = residual / cfg.zone_load_mw

        # ---- DAA price from the merit order + AR residual + spikes
        p = cfg.prices
        price = (
            p.base_price_eur_per_mwh
            + p.residual_slope_eur_per_mwh * residual_norm
            + 55.0 * np.maximum(0.0, residual_norm - 0.55) ** 2  # scarcity convexity
            + _ar1(rng, n, 0.95, 4.0) * vol
        )
        if p.positive_spikes:
            price = price + _spikes(rng, n, p.spikes_per_year, per_year, 20, 900) * (
                0.4 + 0.6 * regime
            )
        if p.negative_spikes:
            price = price - _spikes(rng, n, p.spikes_per_year, per_year, 20, 700) * (
                0.4 + 0.6 * regime
            )
        if not p.allow_negative_prices:
            price = np.maximum(price, 0.0)
        self.daa_qh = np.clip(price, p.minimum_price_eur_per_mwh, p.maximum_price_eur_per_mwh)

        # ---- forecast errors (shared by zone tables and the intraday chain)
        def error(sigma_cf: float, phi: float) -> np.ndarray:
            return _ar1(rng, n, phi, sigma_cf * np.sqrt(1 - phi * phi))

        self.err_da_wind = error(0.055, 0.997) * cfg.zone_wind_capacity_mw
        self.err_id_wind = 0.45 * self.err_da_wind + error(0.02, 0.995) * (
            cfg.zone_wind_capacity_mw
        )
        self.err_da_pv = error(0.03, 0.995) * cfg.zone_pv_capacity_mw * clearsky
        self.err_id_pv = 0.45 * self.err_da_pv + error(0.012, 0.99) * (
            cfg.zone_pv_capacity_mw * clearsky
        )

        # information updates between auctions (fractions of the DA error resolved)
        slope = p.residual_slope_eur_per_mwh / cfg.zone_load_mw
        upd1 = -slope * 0.35 * (self.err_da_wind + self.err_da_pv)
        upd2 = -slope * 0.20 * (self.err_da_wind + self.err_da_pv)
        upd3 = -slope * 0.15 * (self.err_da_wind + self.err_da_pv)

        clip = lambda x: np.clip(  # noqa: E731
            x, p.minimum_price_eur_per_mwh, p.maximum_price_eur_per_mwh
        )
        self.ida1 = clip(self.daa_qh + upd1 + _ar1(rng, n, 0.9, 5.0) * vol)
        self.ida2 = clip(self.ida1 + upd2 + _ar1(rng, n, 0.9, 3.5) * vol)
        self.ida3 = clip(self.ida2 + upd3 + _ar1(rng, n, 0.9, 2.5) * vol)

        # ---- IDC and reBAP
        idc_mid = clip(self.ida3 + _ar1(rng, n, 0.85, 3.0) * vol)
        total_err = (self.err_id_wind + self.err_id_pv) / (
            0.05 * (cfg.zone_wind_capacity_mw + cfg.zone_pv_capacity_mw)
        )
        rebap_term = -35.0 * total_err + _ar1(rng, n, 0.7, 18.0) * vol
        self.rebap = clip(idc_mid + rebap_term)
        self.id1 = clip(idc_mid + 0.35 * rebap_term + rng.normal(0, 2.0, n) * vol)
        self.id3 = clip(idc_mid + 0.15 * rebap_term + rng.normal(0, 1.5, n) * vol)
        self.idfull = clip(idc_mid + rng.normal(0, 1.0, n) * vol)
        self.spread = np.maximum(
            self.cfg.idc.minimum_spread_eur_per_mwh, 1.5 + 2.5 * regime + 0.3 * np.abs(total_err)
        )
        liquidity_shape = 0.6 + 0.4 * np.exp(-((hour - 12.0) ** 2) / 40.0)
        self.volume = rng.lognormal(np.log(60.0), 0.5, n) * liquidity_shape

        self.load, self.wind_mw, self.pv_mw = load, wind_mw, pv_mw
        # offshore proxy: persistent, loosely coupled to onshore conditions
        self.wind_off_mw = (
            _sigmoid(_ar1(rng, n, 0.998, 0.06) + 0.35 * annual) * 0.09 * cfg.zone_wind_capacity_mw
        )


# ------------------------------------------------------------------- database


def generate_synthetic_database(
    cfg: SyntheticMarketConfig,
    markets: MarketsConfig,
    out_path: Path,
    force: bool = False,
) -> Path:
    """Create (or reuse) the synthetic drop-in database at ``out_path``."""
    out_path = Path(out_path)
    digest = config_hash(cfg)
    if out_path.exists() and not force:
        meta = read_metadata(out_path)
        if meta.get("config_hash") == digest and meta.get("schema_version") == SCHEMA_VERSION:
            log.info("reusing synthetic database %s (config hash %s)", out_path, digest)
            return out_path
        log.info("synthetic database outdated (config/schema changed) — regenerating")

    series = SyntheticSeries(cfg, markets)
    grid = series.grid
    local_naive = grid.tz_convert(MARKET_TZ).tz_localize(None)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".building")
    if tmp.exists():
        tmp.unlink()
    with sqlite3.connect(tmp) as con:
        _write_daa(con, series, markets, local_naive)
        _write_ida_wide(con, series)
        _write_idc(con, series)
        _write_rebap(con, series, local_naive)
        _write_zone(con, series, local_naive)
        _write_metadata(con, cfg, digest)
        con.commit()
    tmp.replace(out_path)
    log.info(
        "generated synthetic market database: %s (%d intervals, seed %d)",
        out_path,
        len(grid),
        cfg.random_seed,
    )
    return out_path


def _write_daa(con, series, markets: MarketsConfig, local_naive) -> None:
    switch = markets.daa_quarter_hourly_from
    is_qh_era = local_naive >= switch
    qh = pd.DataFrame({"index": local_naive, "price_eur_per_mwh": series.daa_qh})
    post = qh[is_qh_era]
    pre = qh[~is_qh_era].copy()
    if len(pre):
        # hourly products: mean of the four quarter-hours, one row per local hour
        # (grouping by floored *UTC* time keeps the duplicated autumn hour distinct)
        utc_hour = series.grid[~is_qh_era].floor("1h")
        pre = (
            pre.assign(_h=utc_hour)
            .groupby("_h", sort=True)
            .agg(index=("index", "first"), price_eur_per_mwh=("price_eur_per_mwh", "mean"))
            .reset_index(drop=True)
        )
    frame = pd.concat([pre, post[["index", "price_eur_per_mwh"]]], ignore_index=True)
    frame.to_sql("day_ahead_prices", con, index=False, if_exists="replace")
    con.execute('CREATE INDEX ix_day_ahead_prices_index ON day_ahead_prices ("index")')


def _write_ida_wide(con, series) -> None:
    price_by_market = {"ida1": series.ida1, "ida2": series.ida2, "ida3": series.ida3}
    hours_by_market = {"ida1": range(0, 24), "ida2": range(0, 24), "ida3": range(12, 24)}
    prices = {
        market: pd.Series(values, index=series.grid) for market, values in price_by_market.items()
    }
    local_days = series.grid.tz_convert(MARKET_TZ).normalize().unique()

    for market, hours in hours_by_market.items():
        # column list matches the real wide layout: hour 3 -> '3a' plus a '3b' block
        columns = []
        for h in hours:
            base = "3a" if h == 2 else str(h + 1)
            columns.extend(f"hour {base} q{q}" for q in range(1, 5))
            if h == 2:
                columns.extend(f"hour 3b q{q}" for q in range(1, 5))
        rows = []
        for day in local_days:
            day_naive = day.tz_localize(None)
            row: dict[str, object] = {"Delivery day": day_naive}
            products = _day_products(day_naive)
            long_day = len(products) == 100
            for start_utc, wall in products:
                if wall.hour not in hours:
                    continue
                col = MarketDataStore._ida_column(wall, long_day)
                row[col] = float(prices[market].loc[start_utc])
            rows.append(row)
        frame = pd.DataFrame(rows, columns=["Delivery day", *columns])
        frame.to_sql(f"pan_european_{market}_prices", con, index=False, if_exists="replace")


def _day_products(day_naive: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    start, end = local_day_bounds_utc(day_naive)
    starts = pd.date_range(start, end, freq="15min", inclusive="left")
    return [(s, s.tz_convert(MARKET_TZ)) for s in starts]


def _write_idc(con, series) -> None:
    naive_utc = series.grid.tz_localize(None)
    end_utc = (series.grid + pd.Timedelta(minutes=15)).tz_localize(None)
    parts = []
    for name, values in (("ID1", series.id1), ("ID3", series.id3), ("IDFULL", series.idfull)):
        parts.append(
            pd.DataFrame(
                {
                    "Time": naive_utc,
                    "IndexName": name,
                    "TimeResolution": "15min",
                    "IndexPrice": values,
                    "IndexVolume": series.volume,
                    "DeliveryEnd": end_utc,
                    "Currency": "EUR",
                    "VolumeUnit": "MWh",
                }
            )
        )
    frame = pd.concat(parts, ignore_index=True)
    frame.to_sql("intraday_continuous_indices", con, index=False, if_exists="replace")
    con.execute("CREATE INDEX ix_idc_time ON intraday_continuous_indices (Time, IndexName)")

    if series.cfg.idc.generate_bid_ask:
        mid = series.idfull
        stats = pd.DataFrame(
            {
                "deliverystart": series.grid.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "deliveryend": (series.grid + pd.Timedelta(minutes=15)).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
                "lowprice": mid - series.spread / 2.0,  # bid proxy
                "highprice": mid + series.spread / 2.0,  # ask proxy
                "lastprice": series.id1,
                "weightedaverageprice": mid,
                "currency": "EUR",
                "volumebuy": series.volume / 2.0,
                "volumesell": series.volume / 2.0,
                "volumeunit": "MWH",
            }
        )
        stats.to_sql("intraday_continuous_statistics", con, index=False, if_exists="replace")


def _write_rebap(con, series, local_naive) -> None:
    naive_utc = series.grid.tz_localize(None)
    imb = series.cfg.imbalance
    half = imb.spread_eur_per_mwh / 2.0 if imb.asymmetric else 0.0
    frame = pd.DataFrame(
        {
            "Date": naive_utc,
            "Datum": naive_utc.strftime("%d.%m.%Y"),
            "Zeitzone": "UTC",
            "von": naive_utc.strftime("%H:%M"),
            "bis": (naive_utc + pd.Timedelta(minutes=15)).strftime("%H:%M"),
            "Datenkategorie": "reBAP",
            "Datentyp": "Synthetisch",
            "Einheit": "EUR/MWh",
            "reBAP unterdeckt": series.rebap + half,
            "reBAP ueberdeckt": series.rebap - half,
        }
    )
    frame.to_sql(
        "netztransparenz_nrv_saldo_reBAP_Qualitaetsgesichert",
        con,
        index=False,
        if_exists="replace",
    )
    mirror = pd.DataFrame(
        {
            "index": local_naive,
            "Long": series.rebap,
            "Short": series.rebap,
        }
    )
    mirror.to_sql("imbalance_prices", con, index=False, if_exists="replace")


def _write_zone(con, series, local_naive) -> None:
    actual = pd.DataFrame(
        {
            "index": local_naive,
            "Wind Onshore_Actual Aggregated": series.wind_mw,
            "Wind Offshore_Actual Aggregated": series.wind_off_mw,
            "Solar_Actual Aggregated": series.pv_mw,
        }
    )
    actual.to_sql("generation", con, index=False, if_exists="replace")

    def forecast_frame(err_wind, err_pv):
        return pd.DataFrame(
            {
                "index": local_naive,
                "Solar": np.maximum(0.0, series.pv_mw + err_pv),
                "Wind Offshore": series.wind_off_mw,
                "Wind Onshore": np.maximum(0.0, series.wind_mw + err_wind),
            }
        )

    forecast_frame(series.err_da_wind, series.err_da_pv).to_sql(
        "wind_and_solar_forecast", con, index=False, if_exists="replace"
    )
    forecast_frame(series.err_id_wind, series.err_id_pv).to_sql(
        "intraday_wind_and_solar_forecast", con, index=False, if_exists="replace"
    )


def _write_metadata(con, cfg: SyntheticMarketConfig, digest: str) -> None:
    meta = {
        "synthetic": "true",
        "generator_version": GENERATOR_VERSION,
        "schema_version": SCHEMA_VERSION,
        "config_hash": digest,
        "random_seed": str(cfg.random_seed),
        "start": str(cfg.start.date()),
        "end": str(cfg.end.date()),
        "resolution_minutes": str(cfg.resolution_minutes),
        "bidding_zone": cfg.bidding_zone,
        "calibration": "enabled" if cfg.calibration.enabled else "defaults",
        "created_at": datetime.now().astimezone().isoformat(),
        "git_commit": _git_commit() or "unknown",
        "config_json": cfg.model_dump_json(),
    }
    con.execute("CREATE TABLE synthetic_metadata (key TEXT PRIMARY KEY, value TEXT)")
    con.executemany("INSERT INTO synthetic_metadata VALUES (?, ?)", list(meta.items()))


def read_metadata(db_path: Path) -> dict[str, str]:
    """Metadata of a synthetic database ({} for real/unknown databases)."""
    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as con:
            rows = con.execute("SELECT key, value FROM synthetic_metadata").fetchall()
        return dict(rows)
    except sqlite3.Error:
        return {}


# --------------------------------------------------------------------------
# CONFIG — edit and run as a module (no CLI flags by design)
# --------------------------------------------------------------------------
CONFIG_PATH = Path("configs/synthetic_market.yaml")
FORCE_REGENERATE = False

if __name__ == "__main__":
    from hybrid_vpp.config.models import load_config

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    experiment = load_config(CONFIG_PATH)
    generate_synthetic_database(
        experiment.synthetic_market,
        experiment.markets,
        experiment.data.market_database.synthetic_path,
        force=FORCE_REGENERATE,
    )
