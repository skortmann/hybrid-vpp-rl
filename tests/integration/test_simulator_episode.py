"""Full-episode integration tests against the real market database.

Skipped when the IAEW database is unavailable. Uses the parquet cache built
on first access, so repeated runs are fast.
"""

from pathlib import Path

import pandas as pd
import pytest

from hybrid_vpp.config.models import ExperimentConfig, load_config
from hybrid_vpp.core.timegrid import energy_mwh
from hybrid_vpp.data.site_profiles import load_site_profiles
from hybrid_vpp.data.sqlite_market_data import MarketDataStore
from hybrid_vpp.markets.calendar import EventType
from hybrid_vpp.sim.simulator import (
    AuctionAction,
    DispatchAction,
    IdcAction,
    Simulator,
)
from tests.conftest import requires_real_db

CONFIG = Path(__file__).parents[2] / "configs" / "default.yaml"

pytestmark = requires_real_db


@pytest.fixture(scope="module")
def cfg() -> ExperimentConfig:
    return load_config(CONFIG)


@pytest.fixture(scope="module")
def store(cfg) -> MarketDataStore:
    return MarketDataStore(cfg.data, cfg.markets)


@pytest.fixture(scope="module")
def profiles(cfg, store):
    return load_site_profiles(cfg.data, cfg.site, store)


@pytest.fixture()
def sim(cfg, store, profiles) -> Simulator:
    return Simulator(cfg, store, profiles)


def run_scripted(sim: Simulator, day: str, daa_mw=50.0, days=1) -> Simulator:
    event = sim.start_episode(pd.Timestamp(day), days=days)
    while event is not None:
        if event.type == EventType.DAA_GATE_CLOSURE:
            action = AuctionAction({p: daa_mw for p in event.products})
        elif event.type in (
            EventType.IDA1_GATE_CLOSURE,
            EventType.IDA2_GATE_CLOSURE,
            EventType.IDA3_GATE_CLOSURE,
        ):
            action = AuctionAction()
        elif event.type == EventType.IDC_DECISION:
            action = IdcAction()
        else:
            action = DispatchAction()
        _, event = sim.step(action)
    return sim


class TestAccountingInvariants:
    def test_episode_accounting(self, sim):
        run_scripted(sim, "2025-03-15")
        # each euro exactly once: ledger totals equal their components
        totals = sim.ledger.by_component()
        assert sim.ledger.total() == pytest.approx(sum(totals.values()))
        # market cash equals the sum over individual trade cash flows
        trade_cash = sum(t.cash_flow_eur for t in sim.book.trades)
        market_cash = sum(totals[m] for m in ("daa", "ida1", "ida2", "ida3", "idc"))
        assert market_cash == pytest.approx(trade_cash)
        # transaction costs booked once per trade
        assert totals["transaction_cost"] == pytest.approx(
            -sum(t.transaction_cost_eur for t in sim.book.trades)
        )
        # per-interval deviation identity and single settlement
        assert len(sim.settlements) == 96
        for start, s in sim.settlements.items():
            record = sim.dispatch_records[start]
            delivered = energy_mwh(record.dispatch.grid_power_mw, record.product.duration)
            assert s.delivered_mwh == pytest.approx(delivered)
            assert s.deviation_mwh == pytest.approx(s.delivered_mwh - s.contracted_mwh)
        imbalance_entries = [e for e in sim.ledger.entries if e.component == "imbalance"]
        assert len(imbalance_entries) == 96

    def test_case6_commitment_above_deliverable(self, sim):
        """Market position above physical capability: physics stays limited,
        the deviation is settled exactly once per interval."""
        run_scripted(sim, "2025-03-15", daa_mw=140.0)  # limit is 150 MW cap, X=100
        for record in sim.dispatch_records.values():
            assert record.dispatch.grid_power_mw <= sim.cfg.site.grid.export_limit_mw + 1e-6
        total_dev = sum(s.deviation_mwh for s in sim.settlements.values())
        assert total_dev < -1000.0  # massively short, as expected
        imbalance_entries = [e for e in sim.ledger.entries if e.component == "imbalance"]
        assert len(imbalance_entries) == len(sim.settlements) == 96


class TestMarketTimingIntegration:
    def test_frozen_daa_position_cannot_be_resubmitted(self, sim):
        """After the DAA gate, later auction events reject orders on DAA products."""
        event = sim.start_episode(pd.Timestamp("2025-03-15"))
        assert event.type == EventType.DAA_GATE_CLOSURE
        daa_products = event.products
        _, event = sim.step(AuctionAction({p: 10.0 for p in daa_products}))
        assert event.type == EventType.IDA1_GATE_CLOSURE
        with pytest.raises(ValueError, match="outside auction scope"):
            sim.step(AuctionAction({daa_products[0]: 99.0}))

    def test_missing_ida2_auction_day_yields_no_fills(self, sim):
        """2025-02-04 has no IDA2 clearing prices (auction not held)."""
        event = sim.start_episode(pd.Timestamp("2025-02-04"))
        reports_by_market = {}
        while event is not None:
            if event.type == EventType.DAA_GATE_CLOSURE:
                action = AuctionAction({p: 20.0 for p in event.products})
            elif event.type in (
                EventType.IDA1_GATE_CLOSURE,
                EventType.IDA2_GATE_CLOSURE,
                EventType.IDA3_GATE_CLOSURE,
            ):
                action = AuctionAction({p: 5.0 for p in event.products})
            elif event.type == EventType.IDC_DECISION:
                action = IdcAction()
            else:
                action = DispatchAction()
            result, event = sim.step(action)
            if result.event.market:
                reports_by_market.setdefault(result.event.market, []).extend(
                    result.execution_reports
                )
        ida2 = reports_by_market["ida2"]
        assert ida2 and all(r.reason == "no_clearing_price" for r in ida2)
        assert all(r.filled_mw == 0.0 for r in ida2)
        assert sim.book.turnover_mwh("ida2") == 0.0
        # ida1 filled normally on the same day
        assert sim.book.turnover_mwh("ida1") > 0.0


class TestDstEpisodes:
    """Case 8: grid constraint holds for every actual physical interval."""

    @pytest.mark.parametrize("day,n_intervals", [("2025-03-30", 92), ("2024-10-27", 100)])
    def test_dst_day_episode(self, sim, day, n_intervals):
        run_scripted(sim, day, daa_mw=30.0)
        assert len(sim.dispatch_records) == n_intervals
        assert len(sim.settlements) == n_intervals
        for record in sim.dispatch_records.values():
            assert record.dispatch.grid_power_mw <= sim.cfg.site.grid.export_limit_mw + 1e-6
        starts = sorted(sim.dispatch_records)
        diffs = {b - a for a, b in zip(starts, starts[1:], strict=False)}
        assert diffs == {pd.Timedelta(minutes=15)}  # no skipped/duplicated interval

    def test_hourly_daa_on_25h_day_has_25_products(self, sim):
        event = sim.start_episode(pd.Timestamp("2024-10-27"))
        assert event.type == EventType.DAA_GATE_CLOSURE
        assert len(event.products) == 25
