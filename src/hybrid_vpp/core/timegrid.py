"""Canonical time grid, delivery products, and power/energy conversion.

Conventions (see docs/data_audit.md):

* All internal timestamps are timezone-aware UTC ``pandas.Timestamp``.
* The market-local wall clock (``Europe/Berlin``) is used only at the
  calendar/reporting boundary, never for internal indexing.
* A delivery product is identified by its UTC start and its duration.
  No day is ever assumed to contain exactly 96 local quarter-hours.
* Every power-to-energy conversion in the code base goes through
  :func:`energy_mwh` / :func:`power_mw`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from zoneinfo import ZoneInfo

import pandas as pd

MARKET_TZ = ZoneInfo("Europe/Berlin")
UTC = ZoneInfo("UTC")

#: Canonical internal interval (15-minute delivery periods).
INTERVAL = timedelta(minutes=15)
INTERVAL_HOURS = INTERVAL.total_seconds() / 3600.0


def as_utc(ts: pd.Timestamp | str) -> pd.Timestamp:
    """Return a tz-aware UTC timestamp; naive input is rejected."""
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        raise ValueError(f"naive timestamp {ts!r}: internal timestamps must be tz-aware")
    return ts.tz_convert("UTC")


def local_to_utc(ts: pd.Timestamp | str) -> pd.Timestamp:
    """Interpret a naive timestamp as Europe/Berlin wall time and convert to UTC.

    Ambiguous (autumn DST) times raise: callers that can encounter them must
    resolve ambiguity explicitly at the data layer.
    """
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        return ts.tz_convert("UTC")
    return ts.tz_localize(MARKET_TZ, ambiguous="raise", nonexistent="raise").tz_convert("UTC")


def utc_to_local(ts: pd.Timestamp) -> pd.Timestamp:
    return as_utc(ts).tz_convert(MARKET_TZ)


def duration_hours(duration: timedelta) -> float:
    return duration.total_seconds() / 3600.0


def energy_mwh(power_mw: float, duration: timedelta = INTERVAL) -> float:
    """Convert constant power over ``duration`` to energy: E = P * h."""
    return power_mw * duration_hours(duration)


def power_mw(mwh: float, duration: timedelta = INTERVAL) -> float:
    """Convert energy delivered over ``duration`` to constant power: P = E / h."""
    return mwh / duration_hours(duration)


@dataclass(frozen=True, slots=True, order=True)
class DeliveryProduct:
    """One market delivery product, identified by UTC start and duration."""

    start_utc: pd.Timestamp
    duration: timedelta = INTERVAL

    def __post_init__(self) -> None:
        if self.start_utc.tzinfo is None:
            raise ValueError("DeliveryProduct.start_utc must be tz-aware")
        object.__setattr__(self, "start_utc", self.start_utc.tz_convert("UTC"))

    @property
    def end_utc(self) -> pd.Timestamp:
        return self.start_utc + self.duration

    @property
    def hours(self) -> float:
        return duration_hours(self.duration)

    @property
    def local_start(self) -> pd.Timestamp:
        return self.start_utc.tz_convert(MARKET_TZ)

    def quarter_hours(self) -> list[DeliveryProduct]:
        """Split this product into canonical 15-minute delivery intervals."""
        n, rem = divmod(self.duration, INTERVAL)
        if rem:
            raise ValueError(f"duration {self.duration} is not a multiple of {INTERVAL}")
        return [DeliveryProduct(self.start_utc + i * INTERVAL, INTERVAL) for i in range(int(n))]

    @property
    def id(self) -> str:
        mins = int(self.duration.total_seconds() // 60)
        return f"{self.start_utc.strftime('%Y-%m-%dT%H:%M')}Z/{mins}min"

    def __str__(self) -> str:  # pragma: no cover
        return self.id


def local_day_bounds_utc(day: pd.Timestamp | str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """UTC [start, end) of a Europe/Berlin calendar day (23/24/25 hours long)."""
    d = pd.Timestamp(day)
    if d.tzinfo is not None:
        d = d.tz_convert(MARKET_TZ).tz_localize(None)
    d = d.normalize()
    start = d.tz_localize(MARKET_TZ, ambiguous="raise", nonexistent="raise")
    end = (d + timedelta(days=1)).tz_localize(MARKET_TZ, ambiguous="raise", nonexistent="raise")
    return start.tz_convert("UTC"), end.tz_convert("UTC")


def delivery_intervals_of_local_day(day: pd.Timestamp | str) -> list[DeliveryProduct]:
    """All canonical 15-min delivery intervals of a local calendar day (92/96/100)."""
    start, end = local_day_bounds_utc(day)
    return intervals_between(start, end)


def intervals_between(start_utc: pd.Timestamp, end_utc: pd.Timestamp) -> list[DeliveryProduct]:
    """Canonical 15-min intervals in [start_utc, end_utc)."""
    start_utc, end_utc = as_utc(start_utc), as_utc(end_utc)
    index = pd.date_range(start_utc, end_utc, freq=INTERVAL, inclusive="left")
    return [DeliveryProduct(ts, INTERVAL) for ts in index]


def utc_grid(start_utc: pd.Timestamp, end_utc: pd.Timestamp) -> pd.DatetimeIndex:
    """Canonical 15-min UTC DatetimeIndex over [start, end)."""
    return pd.date_range(as_utc(start_utc), as_utc(end_utc), freq=INTERVAL, inclusive="left")
