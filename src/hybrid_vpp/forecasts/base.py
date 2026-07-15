"""Forecast provider interfaces.

Forecasts are always requested as ``(issue_time, delivery_times)`` — a
forecast matrix indexed only by delivery time could leak later forecast
updates into earlier decisions. Providers must return values that were
(or plausibly would have been) available at ``issue_time``; perfect and
noisy-oracle providers exist for benchmarking/debugging only and are named
accordingly.
"""

from __future__ import annotations

from typing import Protocol

import pandas as pd


class RenewableForecastProvider(Protocol):
    def forecast(self, issue_time: pd.Timestamp, delivery_times: pd.DatetimeIndex) -> pd.DataFrame:
        """Site forecast, MW. Columns: ``wind_mw``, ``pv_mw``; index = delivery_times."""
        ...


class PriceForecastProvider(Protocol):
    def forecast(
        self, market: str, issue_time: pd.Timestamp, delivery_times: pd.DatetimeIndex
    ) -> pd.Series:
        """Price forecast, EUR/MWh, indexed by delivery start (UTC)."""
        ...
