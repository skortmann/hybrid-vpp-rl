"""Market calendar: gate times, product eligibility, DST-safe event streams."""

import pandas as pd
import pytest

from hybrid_vpp.config.models import MarketsConfig
from hybrid_vpp.markets.calendar import EventType, MarketCalendar


@pytest.fixture
def cal() -> MarketCalendar:
    return MarketCalendar(MarketsConfig())


def utc(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


def test_daa_gate_is_noon_local_day_before(cal):
    # winter: 12:00 CET = 11:00 UTC; summer: 12:00 CEST = 10:00 UTC
    assert cal.gate_closure("daa", pd.Timestamp("2025-01-15")) == utc("2025-01-14 11:00")
    assert cal.gate_closure("daa", pd.Timestamp("2025-07-15")) == utc("2025-07-14 10:00")


def test_ida_gate_times(cal):
    day = pd.Timestamp("2025-07-15")
    assert cal.gate_closure("ida1", day) == utc("2025-07-14 13:00")  # 15:00 D-1 CEST
    assert cal.gate_closure("ida2", day) == utc("2025-07-14 20:00")  # 22:00 D-1 CEST
    assert cal.gate_closure("ida3", day) == utc("2025-07-15 08:00")  # 10:00 D CEST


def test_daa_products_hourly_before_switch(cal):
    products = cal.daa_products(pd.Timestamp("2025-01-15"))
    assert len(products) == 24
    assert all(p.hours == 1.0 for p in products)


def test_daa_products_quarter_hourly_after_switch(cal):
    products = cal.daa_products(pd.Timestamp("2025-10-15"))
    assert len(products) == 96
    assert all(p.hours == 0.25 for p in products)


def test_daa_products_on_dst_days(cal):
    assert len(cal.daa_products(pd.Timestamp("2025-03-30"))) == 23  # spring, hourly era
    assert len(cal.daa_products(pd.Timestamp("2024-10-27"))) == 25  # autumn, hourly era
    assert len(cal.daa_products(pd.Timestamp("2026-03-29"))) == 92  # spring, 15-min era


def test_ida3_covers_afternoon_only(cal):
    products = cal.auction_products("ida3", pd.Timestamp("2025-07-15"))
    assert len(products) == 48
    hours = {p.local_start.hour for p in products}
    assert hours == set(range(12, 24))


def test_ida1_covers_full_day(cal):
    assert len(cal.auction_products("ida1", pd.Timestamp("2025-07-15"))) == 96
    assert len(cal.auction_products("ida1", pd.Timestamp("2024-10-27"))) == 100


def test_idc_gate_closure_lead(cal):
    products = cal.auction_products("ida1", pd.Timestamp("2025-07-15"))
    p = products[0]  # delivery 00:00 local = 22:00 UTC on the 14th
    assert cal.idc_gate_closure(p) == p.start_utc - pd.Timedelta(minutes=30)


def test_idc_open_products_respect_gates(cal):
    day = pd.Timestamp("2025-07-15")
    # 23:00 UTC on the 14th = 01:00 local on the 15th: products up to 01:30
    # (exclusive via gate lead) are closed, later ones open
    t = utc("2025-07-14 23:00")
    open_products = cal.idc_open_products(t, [day])
    starts = [p.start_utc for p in open_products]
    assert min(starts) == utc("2025-07-14 23:30")  # 01:30 local
    assert max(starts) == utc("2025-07-15 21:45")
    # before opening time nothing is tradable
    assert cal.idc_open_products(utc("2025-07-14 10:00"), [day]) == []


def test_no_action_can_affect_past_delivery(cal):
    day = pd.Timestamp("2025-07-15")
    t = utc("2025-07-15 12:00")
    for p in cal.idc_open_products(t, [day]):
        assert p.start_utc > t


def test_event_stream_ordering_and_counts(cal):
    events = cal.episode_events([pd.Timestamp("2025-03-15")])
    assert events[0].type == EventType.EPISODE_RESET
    times = [e.sort_key() for e in events[1:]]
    assert times == sorted(times)
    dispatch = [e for e in events if e.type == EventType.PHYSICAL_DISPATCH]
    settle = [e for e in events if e.type == EventType.DELIVERY_SETTLEMENT]
    assert len(dispatch) == len(settle) == 96
    gates = [e for e in events if e.market in ("daa", "ida1", "ida2", "ida3")]
    assert len(gates) == 4


@pytest.mark.parametrize("day,n_qh", [("2025-03-30", 92), ("2024-10-27", 100)])
def test_event_stream_on_dst_days(cal, day, n_qh):
    events = cal.episode_events([pd.Timestamp(day)])
    dispatch = [e for e in events if e.type == EventType.PHYSICAL_DISPATCH]
    assert len(dispatch) == n_qh
    starts = [e.products[0].start_utc for e in dispatch]
    assert starts == sorted(starts)
    assert len(set(starts)) == n_qh  # no interval skipped or duplicated
    diffs = {b - a for a, b in zip(starts, starts[1:], strict=False)}
    assert diffs == {pd.Timedelta(minutes=15)}  # contiguous in UTC


def test_settlement_never_precedes_dispatch(cal):
    events = cal.episode_events([pd.Timestamp("2025-03-15")])
    dispatched = set()
    for e in events:
        if e.type == EventType.PHYSICAL_DISPATCH:
            dispatched.add(e.products[0].start_utc)
        elif e.type == EventType.DELIVERY_SETTLEMENT:
            assert e.products[0].start_utc in dispatched


def test_gate_publication_after_closure(cal):
    day = pd.Timestamp("2025-07-15")
    for market in ("daa", "ida1", "ida2", "ida3"):
        assert cal.publication(market, day) > cal.gate_closure(market, day)


def test_multi_day_episode_interleaves_gates(cal):
    days = [pd.Timestamp("2025-03-15"), pd.Timestamp("2025-03-16")]
    events = cal.episode_events(days)
    # the DAA gate of day 2 (12:00 local on day 1) must come after dispatch
    # of day-1 morning intervals
    daa2 = next(
        e
        for e in events
        if e.type == EventType.DAA_GATE_CLOSURE
        and e.products[0].local_start.tz_localize(None).normalize() == days[1]
    )
    idx = events.index(daa2)
    prior_dispatch = [e for e in events[:idx] if e.type == EventType.PHYSICAL_DISPATCH]
    assert len(prior_dispatch) > 40  # morning of day 1 already dispatched
