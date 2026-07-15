"""Synthetic market database: generation, statistics, reproducibility, time."""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hybrid_vpp.config.models import (
    DataConfig,
    MarketDatabaseConfig,
    MarketsConfig,
    SyntheticMarketConfig,
)
from hybrid_vpp.core.timegrid import MARKET_TZ
from hybrid_vpp.data.schema_manifest import validate_against_manifest
from hybrid_vpp.data.sqlite_market_data import MarketDataStore
from hybrid_vpp.data.synthetic_market import (
    config_hash,
    generate_synthetic_database,
    read_metadata,
)

MARKETS = MarketsConfig()


@pytest.fixture(scope="module")
def small_cfg() -> SyntheticMarketConfig:
    # spans the spring DST day and enough sunny weeks for negative prices
    return SyntheticMarketConfig(start="2025-03-01", end="2025-06-15", random_seed=42)


@pytest.fixture(scope="module")
def db(tmp_path_factory, small_cfg) -> Path:
    out = tmp_path_factory.mktemp("synth") / "synthetic.db"
    return generate_synthetic_database(small_cfg, MARKETS, out)


@pytest.fixture(scope="module")
def store(db, small_cfg, tmp_path_factory) -> MarketDataStore:
    data_cfg = DataConfig(
        market_database=MarketDatabaseConfig(mode="synthetic", synthetic_path=db),
        cache_dir=tmp_path_factory.mktemp("cache"),
    )
    return MarketDataStore(data_cfg, MARKETS, small_cfg)


class TestSchemaCompatibility:
    def test_manifest_validation_passes(self, db):
        assert validate_against_manifest(db) == []

    def test_store_reads_all_series(self, store):
        daa = store.daa_prices()
        assert {"price_eur_per_mwh", "duration_min"} <= set(daa.columns)
        assert str(daa.index.tz) == "UTC"
        for market in ("ida1", "ida2", "ida3"):
            s = store.ida_prices(market)
            assert str(s.index.tz) == "UTC" and s.notna().all()
        idc = store.idc_indices()
        assert {"ID1", "ID3", "IDFULL"} <= set(idc.columns)
        assert store.rebap().notna().all()
        zone = store.zone_renewables()
        assert {"actual_wind_onshore", "fc_da_solar", "fc_id_wind_onshore"} <= set(zone.columns)

    def test_metadata_marks_synthetic(self, db, small_cfg):
        meta = read_metadata(db)
        assert meta["synthetic"] == "true"
        assert meta["random_seed"] == "42"
        assert meta["config_hash"] == config_hash(small_cfg)
        assert "schema_version" in meta and "generator_version" in meta

    def test_ida3_covers_afternoon_only(self, store):
        hours = set(store.ida_prices("ida3").index.tz_convert(MARKET_TZ).hour)
        assert hours == set(range(12, 24))


class TestStatisticalProperties:
    def test_variance_and_autocorrelation(self, store):
        prices = store.daa_prices().price_eur_per_mwh
        assert prices.std() > 5.0
        autocorr = prices.autocorr(lag=1)
        assert autocorr > 0.5  # serially correlated, not white noise

    def test_intraday_seasonality(self, store):
        daa = store.daa_prices()
        hourly = daa.price_eur_per_mwh.groupby(daa.index.tz_convert(MARKET_TZ).hour).mean()
        # overnight cheaper than the evening peak
        assert hourly.loc[3] < hourly.loc[19]
        assert hourly.max() - hourly.min() > 5.0

    def test_weekday_effect(self, store):
        daa = store.daa_prices()
        dow = daa.price_eur_per_mwh.groupby(daa.index.tz_convert(MARKET_TZ).dayofweek).mean()
        assert dow.loc[[5, 6]].mean() < dow.loc[[1, 2, 3]].mean()

    def test_negative_prices_present(self, store):
        prices = store.daa_prices().price_eur_per_mwh
        assert (prices < 0).any()
        assert prices.min() >= -500.0 and prices.max() <= 1500.0

    def test_cross_market_correlation(self, store):
        daa = store.daa_prices()
        qh = daa[daa.duration_min == 15].price_eur_per_mwh
        hourly = daa[daa.duration_min == 60].price_eur_per_mwh.resample("15min").ffill()
        daa_qh = pd.concat([hourly, qh]).sort_index()
        ida1 = store.ida_prices("ida1")
        common = daa_qh.index.intersection(ida1.index)
        assert np.corrcoef(daa_qh[common], ida1[common])[0, 1] > 0.6
        idc = store.idc_indices()
        common = ida1.index.intersection(idc.index)
        assert np.corrcoef(ida1[common], idc.IDFULL[common])[0, 1] > 0.6

    def test_bid_ask_spread_non_negative(self, store):
        stats = store.idc_statistics()
        spread = stats.highprice - stats.lowprice
        assert (spread >= 0.1 - 1e-9).all()

    def test_intraday_forecast_error_smaller_than_day_ahead(self, store):
        zone = store.zone_renewables()
        err_da = (zone.fc_da_wind_onshore - zone.actual_wind_onshore).abs().mean()
        err_id = (zone.fc_id_wind_onshore - zone.actual_wind_onshore).abs().mean()
        assert err_id < err_da


class TestReproducibility:
    def test_same_seed_identical_content(self, small_cfg, tmp_path):
        a = generate_synthetic_database(small_cfg, MARKETS, tmp_path / "a.db", force=True)
        b = generate_synthetic_database(small_cfg, MARKETS, tmp_path / "b.db", force=True)
        for table in (
            "day_ahead_prices",
            "pan_european_ida1_prices",
            "intraday_continuous_indices",
            "netztransparenz_nrv_saldo_reBAP_Qualitaetsgesichert",
        ):
            fa = pd.read_sql(f'SELECT * FROM "{table}"', sqlite3.connect(a))
            fb = pd.read_sql(f'SELECT * FROM "{table}"', sqlite3.connect(b))
            pd.testing.assert_frame_equal(fa, fb)

    def test_different_seed_differs(self, small_cfg, tmp_path):
        other = small_cfg.model_copy(update={"random_seed": 7})
        a = generate_synthetic_database(small_cfg, MARKETS, tmp_path / "a.db", force=True)
        b = generate_synthetic_database(other, MARKETS, tmp_path / "b.db", force=True)
        pa = pd.read_sql("SELECT price_eur_per_mwh FROM day_ahead_prices", sqlite3.connect(a))
        pb = pd.read_sql("SELECT price_eur_per_mwh FROM day_ahead_prices", sqlite3.connect(b))
        assert not np.allclose(pa.to_numpy(), pb.to_numpy())

    def test_config_change_changes_hash(self, small_cfg):
        other = small_cfg.model_copy(update={"random_seed": 7})
        assert config_hash(small_cfg) != config_hash(other)

    def test_cache_reused_and_regenerated(self, small_cfg, tmp_path, caplog):
        out = tmp_path / "c.db"
        generate_synthetic_database(small_cfg, MARKETS, out)
        mtime = out.stat().st_mtime_ns
        generate_synthetic_database(small_cfg, MARKETS, out)  # reuse
        assert out.stat().st_mtime_ns == mtime
        other = small_cfg.model_copy(update={"random_seed": 7})
        generate_synthetic_database(other, MARKETS, out)  # config changed
        assert read_metadata(out)["random_seed"] == "7"


class TestTimeHandling:
    def test_dst_days(self, store):
        ida1 = store.ida_prices("ida1")
        b0 = pd.Timestamp("2025-03-30").tz_localize(MARKET_TZ).tz_convert("UTC")
        b1 = pd.Timestamp("2025-03-31").tz_localize(MARKET_TZ).tz_convert("UTC")
        assert len(ida1[(ida1.index >= b0) & (ida1.index < b1)]) == 92

    def test_utc_ordering_no_duplicates(self, store):
        for series in (
            store.daa_prices().index,
            store.ida_prices("ida1").index,
            store.rebap().index,
        ):
            assert series.is_monotonic_increasing
            assert not series.duplicated().any()

    def test_complete_15min_coverage(self, store):
        reb = store.rebap()
        expected = pd.date_range(reb.index.min(), reb.index.max(), freq="15min")
        assert len(reb) == len(expected)

    def test_leap_day_handled(self, tmp_path):
        cfg = SyntheticMarketConfig(start="2024-02-27", end="2024-03-02", random_seed=1)
        db = generate_synthetic_database(cfg, MARKETS, tmp_path / "leap.db")
        with sqlite3.connect(db) as con:
            n = con.execute(
                "SELECT COUNT(*) FROM netztransparenz_nrv_saldo_reBAP_Qualitaetsgesichert"
            ).fetchone()[0]
        assert n == 4 * 96  # incl. 2024-02-29
