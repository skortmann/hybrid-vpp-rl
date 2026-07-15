"""Read-only access to the IAEW market database, normalized to UTC.

Findings that shape this module (see docs/data_audit.md):

* ENTSO-E tables store timezone-naive **Europe/Berlin** timestamps; the
  autumn DST hour appears as adjacent duplicated rows (first = CEST).
* EEX/EPEX IDA tables are wide (one column per local quarter-hour product,
  ``hour H qQ``); ``hour 3a`` holds the regular 02:00 hour, ``hour 3b`` is
  only filled with the second 02:00 hour of the 25-hour autumn day, and both
  are empty on the 23-hour spring day. The ``*_ordered`` variants in the
  database are corrupt (shifts / dropped DST days) and are **not** used.
* IDC index and statistics tables are UTC.

The source database is opened strictly read-only. Normalized full-history
frames are cached as Parquet under ``DataConfig.cache_dir``.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from hybrid_vpp.config.models import DataConfig, MarketsConfig
from hybrid_vpp.core.timegrid import MARKET_TZ, delivery_intervals_of_local_day

log = logging.getLogger(__name__)

_CACHE_VERSION = "v1"

IDA_TABLES = {
    "ida1": "pan_european_ida1_prices",
    "ida2": "pan_european_ida2_prices",
    "ida3": "pan_european_ida3_prices",
}


def _connect_readonly(db_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def localize_entsoe_index(naive: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Localize a naive Europe/Berlin index (in original row order) to UTC.

    Duplicated wall times from the 25-hour autumn day are resolved by order
    of appearance: first occurrence = CEST (DST), second = CET.
    """
    is_second = pd.Series(naive).duplicated(keep="first").to_numpy()
    localized = naive.tz_localize(MARKET_TZ, ambiguous=~is_second, nonexistent="raise")
    return localized.tz_convert("UTC")


class MarketDataStore:
    """Lazy, cached, UTC-normalized views of all required market series.

    The underlying SQLite file is chosen by the market-database resolver
    (real / synthetic / auto — see :mod:`hybrid_vpp.data.resolver`); the
    provenance is exposed as :attr:`source` for logs and run metadata.
    Parquet caches are namespaced by database identity so real and
    synthetic data can never contaminate each other.
    """

    def __init__(
        self,
        data_cfg: DataConfig,
        markets_cfg: MarketsConfig,
        synthetic_cfg=None,
        required_period: tuple | None = None,
    ) -> None:
        from hybrid_vpp.data.resolver import resolve_market_database

        self.source = resolve_market_database(data_cfg, markets_cfg, synthetic_cfg, required_period)
        self.db_path = Path(self.source.path)
        stat = self.db_path.stat()
        fingerprint = hashlib.sha1(
            f"{self.db_path.resolve()}|{stat.st_size}|{int(stat.st_mtime)}".encode()
        ).hexdigest()[:10]
        self.cache_dir = Path(data_cfg.cache_dir) / f"{self.db_path.stem}-{fingerprint}"
        self.markets_cfg = markets_cfg
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._frames: dict[str, pd.DataFrame] = {}

    @property
    def provenance(self) -> str:
        return self.source.provenance

    # ------------------------------------------------------------ plumbing

    def _cached(self, name: str, builder) -> pd.DataFrame:
        if name in self._frames:
            return self._frames[name]
        path = self.cache_dir / f"{name}_{_CACHE_VERSION}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
        else:
            log.info("building normalized frame %s from %s", name, self.db_path)
            df = builder()
            df.to_parquet(path)
        self._frames[name] = df
        return df

    def _read_sql(self, query: str) -> pd.DataFrame:
        with _connect_readonly(self.db_path) as con:
            return pd.read_sql_query(query, con)

    # ------------------------------------------------------------- series

    def daa_prices(self) -> pd.DataFrame:
        """Day-ahead clearing prices. Index: product start (UTC).

        Columns: ``price_eur_per_mwh``, ``duration_min`` (60 before the SDAC
        15-minute switch, 15 after — verified against per-day row counts).
        """
        return self._cached("daa_prices", self._build_daa)

    def _build_daa(self) -> pd.DataFrame:
        raw = self._read_sql(
            'SELECT "index" AS ts, price_eur_per_mwh FROM day_ahead_prices ORDER BY rowid'
        )
        idx = localize_entsoe_index(pd.DatetimeIndex(pd.to_datetime(raw["ts"])))
        df = pd.DataFrame({"price_eur_per_mwh": raw["price_eur_per_mwh"].to_numpy()}, index=idx)
        df = df[~df.index.duplicated(keep="first")].sort_index()
        switch_local = self.markets_cfg.daa_quarter_hourly_from.tz_localize(MARKET_TZ)
        df["duration_min"] = np.where(df.index < switch_local.tz_convert("UTC"), 60, 15)
        self._validate_daa(df)
        return df

    def _validate_daa(self, df: pd.DataFrame) -> None:
        local_day = df.index.tz_convert(MARKET_TZ).normalize()
        counts = df.groupby([local_day, df["duration_min"]]).size()
        for (day, dur), n in counts.items():
            expected = len(delivery_intervals_of_local_day(day.tz_localize(None)))
            expected = expected if dur == 15 else expected // 4
            if n != expected:
                log.warning("DAA day %s: %d products, expected %d", day.date(), n, expected)

    def ida_prices(self, market: str) -> pd.Series:
        """IDA clearing prices per quarter-hour product. Index: start (UTC)."""
        df = self._cached(f"{market}_prices", lambda: self._build_ida(market))
        return df["price_eur_per_mwh"]

    def _build_ida(self, market: str) -> pd.DataFrame:
        table = IDA_TABLES[market]
        raw = self._read_sql(f'SELECT * FROM "{table}"')
        raw.columns = [c.strip().lower() for c in raw.columns]
        days = pd.to_datetime(raw["delivery day"]).dt.normalize()

        starts: list[pd.Timestamp] = []
        prices: list[float] = []
        for row_idx, day in days.items():
            row = raw.loc[row_idx]
            products = delivery_intervals_of_local_day(day)
            long_day = len(products) == 100
            for p in products:
                wall = p.local_start
                col = self._ida_column(wall, long_day)
                if col not in raw.columns:
                    continue  # e.g. IDA3 tables only carry hours 13-24
                price = row[col]
                if pd.notna(price):
                    starts.append(p.start_utc)
                    prices.append(float(price))
        df = pd.DataFrame(
            {"price_eur_per_mwh": prices}, index=pd.DatetimeIndex(starts, name="start_utc")
        ).sort_index()
        dup = df.index.duplicated(keep="first")
        if dup.any():
            log.warning("%s: dropping %d duplicated products", market, int(dup.sum()))
            df = df[~dup]
        return df

    @staticmethod
    def _ida_column(wall: pd.Timestamp, long_day: bool) -> str:
        """Wide-table column for a local product start (DST-aware)."""
        quarter = wall.minute // 15 + 1
        if wall.hour == 2:
            # 'hour 3b' exists only as the second 02:00 hour of the autumn day
            is_second = long_day and wall.dst() == timedelta(0)
            return f"hour 3{'b' if is_second else 'a'} q{quarter}"
        return f"hour {wall.hour + 1} q{quarter}"

    def idc_indices(self) -> pd.DataFrame:
        """IDC indices per 15-min product. Index: delivery start (UTC).

        Columns: ``ID1``, ``ID3``, ``IDFULL`` (EUR/MWh) and
        ``volume_ID1|ID3|IDFULL`` (MWh traded within the index window).
        """
        return self._cached("idc_indices", self._build_idc_indices)

    def _build_idc_indices(self) -> pd.DataFrame:
        raw = self._read_sql(
            "SELECT Time AS ts, IndexName, IndexPrice, IndexVolume "
            "FROM intraday_continuous_indices WHERE TimeResolution = '15min'"
        )
        raw["ts"] = pd.to_datetime(raw["ts"]).dt.tz_localize("UTC")
        price = raw.pivot_table(index="ts", columns="IndexName", values="IndexPrice")
        volume = raw.pivot_table(index="ts", columns="IndexName", values="IndexVolume")
        volume.columns = [f"volume_{c}" for c in volume.columns]
        df = price.join(volume).sort_index()
        df.index.name = "start_utc"
        return df

    def idc_statistics(self) -> pd.DataFrame:
        """Per-15-min-product IDC session statistics (UTC): low/high/VWAP/last, volumes."""
        return self._cached("idc_statistics", self._build_idc_statistics)

    def _build_idc_statistics(self) -> pd.DataFrame:
        raw = self._read_sql(
            "SELECT deliverystart, deliveryend, lowprice, highprice, lastprice, "
            "weightedaverageprice, volumebuy, volumesell FROM intraday_continuous_statistics"
        )
        start = pd.to_datetime(raw["deliverystart"], utc=True, format="ISO8601")
        end = pd.to_datetime(raw["deliveryend"], utc=True, format="ISO8601")
        minutes = (end - start).dt.total_seconds() / 60.0
        qh = raw[(minutes > 13) & (minutes < 17)].copy()
        qh.index = pd.DatetimeIndex(start[(minutes > 13) & (minutes < 17)], name="start_utc")
        qh = qh.drop(columns=["deliverystart", "deliveryend"]).sort_index()
        return qh[~qh.index.duplicated(keep="first")]

    def rebap(self) -> pd.Series:
        """German imbalance price (reBAP, single price), 15 min, UTC index.

        Primary: quality-assured Netztransparenz series; extended with the
        ENTSO-E mirror where the primary has not been published yet.
        """
        return self._cached("rebap", self._build_rebap)["rebap_eur_per_mwh"]

    def _build_rebap(self) -> pd.DataFrame:
        nt = self._read_sql(
            'SELECT Date AS ts, "reBAP unterdeckt" AS price '
            "FROM netztransparenz_nrv_saldo_reBAP_Qualitaetsgesichert"
        )
        nt_idx = pd.DatetimeIndex(pd.to_datetime(nt["ts"])).tz_localize("UTC")
        primary = pd.Series(nt["price"].to_numpy(), index=nt_idx).sort_index()
        primary = primary[~primary.index.duplicated(keep="first")]

        en = self._read_sql(
            'SELECT "index" AS ts, Long AS price FROM imbalance_prices ORDER BY rowid'
        )
        en_idx = localize_entsoe_index(pd.DatetimeIndex(pd.to_datetime(en["ts"])))
        mirror = pd.Series(en["price"].to_numpy(), index=en_idx).sort_index()
        mirror = mirror[~mirror.index.duplicated(keep="first")]

        combined = primary.combine_first(mirror)
        return pd.DataFrame({"rebap_eur_per_mwh": combined}).rename_axis("start_utc")

    def zone_renewables(self) -> pd.DataFrame:
        """DE-LU zone-level renewable MW series, 15 min, UTC index.

        Columns: ``{actual,fc_da,fc_id}_{wind_onshore,wind_offshore,solar}``.
        Used for the synthetic site-profile fallback and the forecast-error
        model — never as direct site data.
        """
        return self._cached("zone_renewables", self._build_zone_renewables)

    def _build_zone_renewables(self) -> pd.DataFrame:
        spec = {
            "actual": (
                "generation",
                {
                    '"Wind Onshore_Actual Aggregated"': "wind_onshore",
                    '"Wind Offshore_Actual Aggregated"': "wind_offshore",
                    '"Solar_Actual Aggregated"': "solar",
                },
            ),
            "fc_da": (
                "wind_and_solar_forecast",
                {
                    '"Wind Onshore"': "wind_onshore",
                    '"Wind Offshore"': "wind_offshore",
                    '"Solar"': "solar",
                },
            ),
            "fc_id": (
                "intraday_wind_and_solar_forecast",
                {
                    '"Wind Onshore"': "wind_onshore",
                    '"Wind Offshore"': "wind_offshore",
                    '"Solar"': "solar",
                },
            ),
        }
        parts = []
        for prefix, (table, cols) in spec.items():
            sel = ", ".join(f"{src} AS {prefix}_{dst}" for src, dst in cols.items())
            raw = self._read_sql(f'SELECT "index" AS ts, {sel} FROM {table} ORDER BY rowid')
            idx = localize_entsoe_index(pd.DatetimeIndex(pd.to_datetime(raw["ts"])))
            part = raw.drop(columns="ts").set_index(idx).sort_index()
            parts.append(part[~part.index.duplicated(keep="first")])
        df = pd.concat(parts, axis=1).rename_axis("start_utc")
        return df
