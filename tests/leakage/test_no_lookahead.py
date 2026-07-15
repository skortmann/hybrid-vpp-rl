"""Leakage tests: no observation may contain future information.

These tests pin the publication-time guards and forecast construction; the
environment-level test verifies that price observations at the DAA gate do
not equal the realized (not yet published) clearing prices.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hybrid_vpp.config.models import MarketsConfig, load_config
from hybrid_vpp.forecasts.price import HistoricalPriceView, SeasonalNaivePriceForecast
from hybrid_vpp.forecasts.renewable import PersistenceForecast
from hybrid_vpp.markets.calendar import MarketCalendar
from tests.conftest import REAL_DB_AVAILABLE

CONFIG = Path(__file__).parents[2] / "configs" / "default.yaml"


@pytest.fixture(scope="module")
def calendar() -> MarketCalendar:
    return MarketCalendar(MarketsConfig())


class TestPublicationGuards:
    def make_view(self, calendar) -> HistoricalPriceView:
        # synthetic hourly DAA prices for two delivery days
        idx = pd.date_range("2025-03-14 23:00", periods=48, freq="1h", tz="UTC")
        return HistoricalPriceView(calendar, {"daa": pd.Series(range(48), index=idx)})

    def test_price_invisible_before_publication(self, calendar):
        view = self.make_view(calendar)
        # 2025-03-15 delivery: gate 12:00 CET = 11:00 UTC, publication +60 min = 12:00 UTC
        before = view.visible("daa", pd.Timestamp("2025-03-14 11:59", tz="UTC"))
        after = view.visible("daa", pd.Timestamp("2025-03-14 12:01", tz="UTC"))
        day1 = pd.Timestamp("2025-03-15 10:00", tz="UTC")
        assert day1 not in before.index
        assert day1 in after.index

    def test_next_day_prices_stay_hidden(self, calendar):
        view = self.make_view(calendar)
        t = pd.Timestamp("2025-03-14 14:00", tz="UTC")  # after day-1 publication
        visible = view.visible("daa", t)
        assert visible.index.max() < pd.Timestamp("2025-03-15 23:00", tz="UTC")
        # day-2 delivery (2025-03-16) publishes on the 15th only
        assert not (visible.index >= pd.Timestamp("2025-03-15 23:00", tz="UTC")).any()

    def test_seasonal_naive_uses_only_published_prices(self, calendar):
        view = self.make_view(calendar)
        fc = SeasonalNaivePriceForecast(view)
        delivery = pd.DatetimeIndex([pd.Timestamp("2025-03-16 10:00", tz="UTC")])
        # at issue 2025-03-15 11:00 UTC, the price of 2025-03-15 10:00 (published
        # 2025-03-14 13:00) is the 1-day lag -> allowed
        value = fc.forecast("daa", pd.Timestamp("2025-03-15 11:00", tz="UTC"), delivery)
        series = view.series["daa"]
        assert value.iloc[0] == series[pd.Timestamp("2025-03-15 10:00", tz="UTC")]


class TestForecastConstruction:
    def make_profiles(self) -> pd.DataFrame:
        idx = pd.date_range("2025-03-10", periods=4 * 96, freq="15min", tz="UTC")
        rng = np.random.default_rng(0)
        return pd.DataFrame(
            {
                "wind_avail_mw": rng.uniform(0, 70, len(idx)),
                "pv_avail_mw": rng.uniform(0, 50, len(idx)),
            },
            index=idx,
        )

    def test_persistence_never_reads_at_or_after_issue(self):
        profiles = self.make_profiles()
        fc = PersistenceForecast(profiles)
        issue = pd.Timestamp("2025-03-12 10:00", tz="UTC")
        delivery = pd.date_range("2025-03-13 12:00", periods=8, freq="15min", tz="UTC")
        result = fc.forecast(issue, delivery)
        # 24h-lagged sources (2025-03-12 12:00+) lie after the issue time,
        # so persistence must fall back to older lags (48h)
        expected = profiles["wind_avail_mw"].reindex(delivery - pd.Timedelta("48h"))
        assert np.allclose(result["wind_mw"].to_numpy(), expected.to_numpy())


@pytest.mark.skipif(not REAL_DB_AVAILABLE, reason="market database not available")
class TestEnvironmentLeakage:
    def test_daa_gate_observation_excludes_todays_clearing_prices(self):
        from hybrid_vpp.envs.hybrid_vpp_env import HybridVppEnv

        cfg = load_config(CONFIG)
        env = HybridVppEnv(cfg, split="train")
        obs, info = env.reset(seed=0, options={"day": "2025-03-15"})
        assert info["event_type"] == "DAA_GATE_CLOSURE"

        n = env.layout.n_slots
        # price block is the third per-slot array (see ObservationBuilder)
        from hybrid_vpp.envs.observations import N_SCALARS, PRICE_SCALE

        price_obs = obs[N_SCALARS + 2 * n : N_SCALARS + 3 * n] * PRICE_SCALE
        realized = env.sim.store.daa_prices()["price_eur_per_mwh"]
        w0 = env._window_start
        realized_day = realized.loc[w0 : w0 + pd.Timedelta(hours=23)]
        # expand hourly products to quarter-hours for comparison
        realized_qh = realized_day.resample("15min").ffill().reindex(env.obs_builder._slot_times)
        valid = realized_qh.notna().to_numpy()
        # the observed prices must NOT match the still-unpublished clearing prices
        assert not np.allclose(price_obs[valid], realized_qh.to_numpy()[valid], atol=0.5)

    def test_split_boundaries_respected(self):
        from hybrid_vpp.envs.hybrid_vpp_env import HybridVppEnv

        cfg = load_config(CONFIG)
        for split in ("train", "val", "test"):
            env = HybridVppEnv(cfg, split=split)
            s = cfg.split
            lo = {"train": s.train_start, "val": s.val_start, "test": s.test_start}[split]
            hi = {"train": s.train_end, "val": s.val_end, "test": s.test_end}[split]
            assert env.valid_days.min() >= lo
            assert env.valid_days.max() < hi
