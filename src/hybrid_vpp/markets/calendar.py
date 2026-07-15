"""Market calendar: event stream with DST-safe gate closures.

All market times are configured as Europe/Berlin wall-clock times and
converted to UTC here — nowhere else in the code base. The calendar produces
the ordered stream of decision events for an episode; the simulator enforces
that an event may only touch its own eligible products.

Event ordering at identical timestamps is deterministic via a fixed priority
(settlement of a past interval < auction gates < IDC < physical dispatch),
so e.g. the 15:00 IDA1 gate closure is always processed before an IDC
decision scheduled at 15:00.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from enum import IntEnum

import pandas as pd

from hybrid_vpp.config.models import AuctionSessionConfig, MarketsConfig
from hybrid_vpp.core.timegrid import (
    INTERVAL,
    MARKET_TZ,
    DeliveryProduct,
    intervals_between,
    local_day_bounds_utc,
)


class EventType(IntEnum):
    """Priority-ordered event types (lower value = processed first at a tie)."""

    EPISODE_RESET = 0
    DELIVERY_SETTLEMENT = 1
    DAA_GATE_CLOSURE = 2
    IDA1_GATE_CLOSURE = 3
    IDA2_GATE_CLOSURE = 4
    IDA3_GATE_CLOSURE = 5
    IDC_DECISION = 6
    PHYSICAL_DISPATCH = 7


AUCTION_EVENTS = {
    "daa": EventType.DAA_GATE_CLOSURE,
    "ida1": EventType.IDA1_GATE_CLOSURE,
    "ida2": EventType.IDA2_GATE_CLOSURE,
    "ida3": EventType.IDA3_GATE_CLOSURE,
}


@dataclass(frozen=True, slots=True)
class MarketEvent:
    time_utc: pd.Timestamp
    type: EventType
    #: products this event may act on (empty for reset/dispatch/settlement)
    products: tuple[DeliveryProduct, ...] = ()
    #: auction results become observable at this time (auction gates only)
    publication_utc: pd.Timestamp | None = None
    #: market key ("daa", "ida1", ..., "idc") for auction/idc events
    market: str | None = None

    def sort_key(self) -> tuple:
        return (self.time_utc, int(self.type))


def _local_time_on(day: pd.Timestamp, wall: object) -> pd.Timestamp:
    """Attach a local wall time to a local calendar day, DST-safe.

    Gate times like 12:00/15:00/22:00 never fall into DST transitions
    (transitions happen 02:00/03:00); ambiguous input raises loudly.
    """
    naive = pd.Timestamp.combine(pd.Timestamp(day).normalize(), wall)
    return naive.tz_localize(MARKET_TZ, ambiguous="raise", nonexistent="raise").tz_convert("UTC")


class MarketCalendar:
    """Builds per-delivery-day product lists and episode event streams."""

    def __init__(self, cfg: MarketsConfig) -> None:
        self.cfg = cfg

    # ------------------------------------------------------------- products

    def daa_products(self, delivery_day: pd.Timestamp) -> list[DeliveryProduct]:
        """DAA products of a local delivery day at their native resolution.

        Hourly products before the SDAC 15-minute switch (23/24/25 products
        on DST days), quarter-hourly afterwards (92/96/100).
        """
        start, end = local_day_bounds_utc(delivery_day)
        if pd.Timestamp(delivery_day).normalize() >= self.cfg.daa_quarter_hourly_from:
            return intervals_between(start, end)
        hours = pd.date_range(start, end, freq="1h", inclusive="left")
        return [DeliveryProduct(ts, timedelta(hours=1)) for ts in hours]

    def auction_products(self, market: str, delivery_day: pd.Timestamp) -> list[DeliveryProduct]:
        """Quarter-hour products covered by an auction for a local delivery day."""
        if market == "daa":
            return self.daa_products(delivery_day)
        session: AuctionSessionConfig = getattr(self.cfg, market)
        h0, h1 = session.delivery_hours
        day = pd.Timestamp(delivery_day).normalize()
        start_local = day + timedelta(hours=h0)
        end_local = day + timedelta(hours=h1)
        start = start_local.tz_localize(MARKET_TZ, ambiguous="raise").tz_convert("UTC")
        end = end_local.tz_localize(MARKET_TZ, ambiguous="raise").tz_convert("UTC")
        return intervals_between(start, end)

    def gate_closure(self, market: str, delivery_day: pd.Timestamp) -> pd.Timestamp:
        session: AuctionSessionConfig = getattr(self.cfg, market)
        gate_day = pd.Timestamp(delivery_day).normalize() + timedelta(days=session.gate_day_offset)
        return _local_time_on(gate_day, session.gate_closure_local)

    def publication(self, market: str, delivery_day: pd.Timestamp) -> pd.Timestamp:
        session: AuctionSessionConfig = getattr(self.cfg, market)
        return self.gate_closure(market, delivery_day) + timedelta(
            minutes=session.publication_delay_min
        )

    def idc_opening(self, delivery_day: pd.Timestamp) -> pd.Timestamp:
        """Products of local day D open for continuous trading on D-1."""
        prev = pd.Timestamp(delivery_day).normalize() - timedelta(days=1)
        return _local_time_on(prev, self.cfg.idc.opening_local_time)

    def idc_gate_closure(self, product: DeliveryProduct) -> pd.Timestamp:
        return product.start_utc - self.cfg.idc.gate_closure_lead

    def idc_open_products(
        self, t: pd.Timestamp, delivery_days: list[pd.Timestamp]
    ) -> list[DeliveryProduct]:
        """All quarter-hour products tradable continuously at time ``t``."""
        out: list[DeliveryProduct] = []
        for day in delivery_days:
            if t < self.idc_opening(day):
                continue
            start, end = local_day_bounds_utc(day)
            for p in intervals_between(start, end):
                if t <= self.idc_gate_closure(p):
                    out.append(p)
        return out

    # ---------------------------------------------------------------- events

    def episode_events(self, delivery_days: list[pd.Timestamp]) -> list[MarketEvent]:
        """Ordered event stream for consecutive local delivery days.

        The stream spans from the first DAA gate (noon before the first
        delivery day) to the settlement of the last delivery interval.
        """
        days = [pd.Timestamp(d).normalize() for d in delivery_days]
        events: list[MarketEvent] = []

        for day in days:
            for market, etype in AUCTION_EVENTS.items():
                session: AuctionSessionConfig = getattr(self.cfg, market)
                if not session.enabled:
                    continue
                events.append(
                    MarketEvent(
                        time_utc=self.gate_closure(market, day),
                        type=etype,
                        products=tuple(self.auction_products(market, day)),
                        publication_utc=self.publication(market, day),
                        market=market,
                    )
                )

        # physical dispatch + settlement on the canonical 15-min grid
        first_start, _ = local_day_bounds_utc(days[0])
        _, last_end = local_day_bounds_utc(days[-1])
        intervals = intervals_between(first_start, last_end)
        for p in intervals:
            events.append(MarketEvent(p.start_utc, EventType.PHYSICAL_DISPATCH, (p,)))
            events.append(MarketEvent(p.end_utc, EventType.DELIVERY_SETTLEMENT, (p,)))

        # IDC decisions on the configured cadence, from first opening to last gate
        if self.cfg.idc.enabled:
            t0 = min(self.idc_opening(d) for d in days)
            t1 = max(self.idc_gate_closure(p) for p in intervals)
            freq = self.cfg.idc.decision_frequency
            t = t0.ceil(pd.Timedelta(INTERVAL))
            while t <= t1:
                products = self.idc_open_products(t, days)
                if products:
                    events.append(
                        MarketEvent(t, EventType.IDC_DECISION, tuple(products), market="idc")
                    )
                t += pd.Timedelta(freq)

        events.sort(key=MarketEvent.sort_key)
        first = events[0].time_utc
        return [MarketEvent(first, EventType.EPISODE_RESET), *events]


@dataclass
class EpisodeClock:
    """Tracks the current position in an episode's event stream."""

    events: list[MarketEvent]
    cursor: int = 0
    _now: pd.Timestamp | None = field(default=None, init=False)

    @property
    def now(self) -> pd.Timestamp:
        return self.events[self.cursor].time_utc if self._now is None else self._now

    @property
    def current(self) -> MarketEvent:
        return self.events[self.cursor]

    @property
    def done(self) -> bool:
        return self.cursor >= len(self.events)

    def advance(self) -> MarketEvent | None:
        self.cursor += 1
        return None if self.done else self.events[self.cursor]
