"""Price forecast providers (per market).

All providers guard against look-ahead: a historical price enters a
forecast only if its publication time lies strictly before the issue time.
Publication is approximated per market as *gate closure + publication
delay* of the delivery day (see MarketCalendar); IDC indices are treated as
published at delivery end.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hybrid_vpp.core.timegrid import MARKET_TZ
from hybrid_vpp.markets.calendar import MarketCalendar


class HistoricalPriceView:
    """As-of view on realized market prices with publication-time guards."""

    def __init__(self, calendar: MarketCalendar, price_series: dict[str, pd.Series]) -> None:
        self.calendar = calendar
        self.series = price_series  # market -> series indexed by delivery start (UTC)
        self._published: dict[str, pd.DatetimeIndex] = {}

    def _publication_index(self, market: str) -> pd.DatetimeIndex:
        """Publication time of each entry of the market's series (cached)."""
        if market not in self._published:
            s = self.series[market]
            if market == "idc":
                published = s.index  # indices publish continuously during delivery
            else:
                days = s.index.tz_convert(MARKET_TZ).normalize().tz_localize(None)
                by_day = {d: self.calendar.publication(market, d) for d in days.unique()}
                published = pd.DatetimeIndex([by_day[d] for d in days])
            self._published[market] = published
        return self._published[market]

    def visible(self, market: str, issue_time: pd.Timestamp) -> pd.Series:
        """All realized prices of `market` published strictly before `issue_time`."""
        s = self.series[market]
        return s[self._publication_index(market) < issue_time]


class SeasonalNaivePriceForecast:
    """Price of the same product 1 day earlier (fallback: 2, 7 days)."""

    def __init__(self, view: HistoricalPriceView) -> None:
        self.view = view

    def forecast(self, market, issue_time, delivery_times) -> pd.Series:
        visible = self.view.visible(market, issue_time)
        out = pd.Series(np.nan, index=delivery_times)
        for days in (1, 2, 7):
            lagged = visible.reindex(delivery_times - pd.Timedelta(days=days))
            out = out.fillna(pd.Series(lagged.to_numpy(), index=delivery_times))
        return out.ffill().fillna(0.0)


class RollingMeanPriceForecast:
    """Mean of the same product over the last `window_days` published days."""

    def __init__(self, view: HistoricalPriceView, window_days: int = 7) -> None:
        self.view = view
        self.window_days = window_days

    def forecast(self, market, issue_time, delivery_times) -> pd.Series:
        visible = self.view.visible(market, issue_time)
        stack = pd.DataFrame(
            {
                d: visible.reindex(delivery_times - pd.Timedelta(days=d)).to_numpy()
                for d in range(1, self.window_days + 1)
            },
            index=delivery_times,
        )
        return stack.mean(axis=1).ffill().fillna(0.0)


class PerfectPriceForecast:
    """Realized prices. Upper-bound benchmarking ONLY."""

    is_perfect_foresight = True

    def __init__(self, price_series: dict[str, pd.Series]) -> None:
        self.series = price_series

    def forecast(self, market, issue_time, delivery_times) -> pd.Series:
        return self.series[market].reindex(delivery_times).ffill().fillna(0.0)


def build_price_forecaster(mode: str, calendar: MarketCalendar, price_series: dict[str, pd.Series]):
    if mode == "perfect":
        return PerfectPriceForecast(price_series)
    view = HistoricalPriceView(calendar, price_series)
    if mode == "seasonal_naive":
        return SeasonalNaivePriceForecast(view)
    if mode == "rolling_mean":
        return RollingMeanPriceForecast(view)
    raise ValueError(f"unknown price forecast mode {mode!r}")
