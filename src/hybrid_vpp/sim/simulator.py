"""Deterministic event-driven simulator of the hybrid VPP.

The simulator walks the market-calendar event stream of an episode and
maintains the commercial and physical state:

* auction gates execute submitted orders at historical clearing prices and
  freeze them (append-only position book),
* IDC decisions execute price-taker fills against historical indices,
* physical-dispatch events project requested dispatch onto the feasible set
  (grid limit, SoC, curtailment bounds) and step the battery,
* settlement events reconcile delivered vs. contracted energy at the
  imbalance price and book degradation / curtailment costs.

Only *decision* events (gates, IDC, dispatch) are exposed to controllers;
settlement is processed automatically. Every euro flows through the ledger
exactly once: market cash at execution time, settlement adjustments at
delivery (documented reward-timing choice).

The simulator itself is controller-agnostic and free of RL dependencies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from hybrid_vpp.assets.battery import Battery, BatteryInterval
from hybrid_vpp.assets.feasibility import FeasibleDispatch, project_dispatch
from hybrid_vpp.config.models import ExperimentConfig
from hybrid_vpp.core.timegrid import DeliveryProduct, energy_mwh
from hybrid_vpp.data.sqlite_market_data import MarketDataStore
from hybrid_vpp.markets.calendar import AUCTION_EVENTS, EventType, MarketCalendar, MarketEvent
from hybrid_vpp.markets.execution import (
    ExecutionReport,
    execute_auction_orders,
    execute_idc_orders,
)
from hybrid_vpp.markets.positions import Ledger, PositionBook
from hybrid_vpp.markets.settlement import ImbalanceSettlement, SettlementResult

log = logging.getLogger(__name__)

DECISION_EVENTS = frozenset(
    {
        EventType.DAA_GATE_CLOSURE,
        EventType.IDA1_GATE_CLOSURE,
        EventType.IDA2_GATE_CLOSURE,
        EventType.IDA3_GATE_CLOSURE,
        EventType.IDC_DECISION,
        EventType.PHYSICAL_DISPATCH,
    }
)


@dataclass
class AuctionAction:
    """Signed MW per eligible product (+ sell / - buy)."""

    orders: dict[DeliveryProduct, float] = field(default_factory=dict)


@dataclass
class IdcAction:
    """Signed MW per still-open product (+ sell / - buy)."""

    orders: dict[DeliveryProduct, float] = field(default_factory=dict)


@dataclass
class DispatchAction:
    """Requested physical dispatch for the current quarter-hour."""

    bess_power_mw: float = 0.0  # + discharge / - charge
    wind_curtail_mw: float = 0.0
    pv_curtail_mw: float = 0.0


Action = AuctionAction | IdcAction | DispatchAction


@dataclass(frozen=True, slots=True)
class DispatchRecord:
    product: DeliveryProduct
    wind_avail_mw: float
    pv_avail_mw: float
    dispatch: FeasibleDispatch
    battery: BatteryInterval


@dataclass
class StepResult:
    """Everything that happened while processing one decision event."""

    event: MarketEvent
    execution_reports: list[ExecutionReport] = field(default_factory=list)
    dispatch_record: DispatchRecord | None = None
    settlements: list[SettlementResult] = field(default_factory=list)
    cash_eur: float = 0.0  # sum of ledger entries booked during this step


class Simulator:
    def __init__(
        self,
        cfg: ExperimentConfig,
        store: MarketDataStore,
        site_profiles: pd.DataFrame,
    ) -> None:
        self.cfg = cfg
        self.store = store
        self.profiles = site_profiles
        self.calendar = MarketCalendar(cfg.markets)
        self.battery = Battery(cfg.site.battery)
        self.settlement = ImbalanceSettlement(
            cfg.markets.imbalance,
            rebap=store.rebap(),
            reference_prices=self._reference_prices()
            if cfg.markets.imbalance.model != "rebap"
            else None,
        )
        self._auction_prices = {
            "daa": store.daa_prices()["price_eur_per_mwh"],
            "ida1": store.ida_prices("ida1"),
            "ida2": store.ida_prices("ida2"),
            "ida3": store.ida_prices("ida3"),
        }
        self._idc_indices = store.idc_indices()

        # episode state
        self.book = PositionBook()
        self.ledger = Ledger()
        self.events: list[MarketEvent] = []
        self.cursor = 0
        self.dispatch_records: dict[pd.Timestamp, DispatchRecord] = {}
        self.settlements: dict[pd.Timestamp, SettlementResult] = {}
        self.delivery_days: list[pd.Timestamp] = []

    def _reference_prices(self) -> pd.Series:
        """Quarter-hour DA reference prices for stylized imbalance models."""
        daa = self.store.daa_prices()
        hourly = daa[daa.duration_min == 60]["price_eur_per_mwh"]
        qh = daa[daa.duration_min == 15]["price_eur_per_mwh"]
        expanded = hourly.resample("15min").ffill() if not hourly.empty else hourly
        return pd.concat([expanded, qh]).sort_index()

    # ----------------------------------------------------------- episode API

    def start_episode(self, first_delivery_day: pd.Timestamp, days: int = 1) -> MarketEvent:
        """Reset state and return the first decision event."""
        day0 = pd.Timestamp(first_delivery_day).normalize()
        self.delivery_days = [day0 + pd.Timedelta(days=i) for i in range(days)]
        self.events = self.calendar.episode_events(self.delivery_days)
        self.cursor = 0
        self.book = PositionBook()
        self.ledger = Ledger()
        self.battery.reset()
        self.dispatch_records = {}
        self.settlements = {}
        return self._advance_to_decision()

    @property
    def current_event(self) -> MarketEvent | None:
        return self.events[self.cursor] if self.cursor < len(self.events) else None

    @property
    def done(self) -> bool:
        return self.cursor >= len(self.events)

    def step(self, action: Action) -> tuple[StepResult, MarketEvent | None]:
        """Process the current decision event, then auto-process bookkeeping
        events up to (and excluding) the next decision event.

        Returns the step result and the next decision event (None when the
        episode is over).
        """
        event = self.current_event
        if event is None:
            raise RuntimeError("episode is over — call start_episode first")
        if event.type not in DECISION_EVENTS:
            raise RuntimeError(f"internal error: {event.type.name} is not a decision event")

        ledger_mark = len(self.ledger)
        result = StepResult(event=event)

        if event.type == EventType.PHYSICAL_DISPATCH:
            if not isinstance(action, DispatchAction):
                raise TypeError(f"{event.type.name} requires DispatchAction, got {type(action)}")
            result.dispatch_record = self._dispatch(event, action)
        elif event.type == EventType.IDC_DECISION:
            if not isinstance(action, IdcAction):
                raise TypeError(f"{event.type.name} requires IdcAction, got {type(action)}")
            result.execution_reports = self._execute_idc(event, action)
        else:
            if not isinstance(action, AuctionAction):
                raise TypeError(f"{event.type.name} requires AuctionAction, got {type(action)}")
            result.execution_reports = self._execute_auction(event, action)

        self.cursor += 1
        next_event = self._advance_to_decision(result)
        result.cash_eur = sum(e.amount_eur for e in self.ledger.entries_since(ledger_mark))
        return result, next_event

    def _advance_to_decision(self, result: StepResult | None = None) -> MarketEvent | None:
        while not self.done:
            event = self.events[self.cursor]
            if event.type in DECISION_EVENTS:
                return event
            if event.type == EventType.DELIVERY_SETTLEMENT:
                settlement = self._settle(event.products[0], event.time_utc)
                if result is not None:
                    result.settlements.append(settlement)
            self.cursor += 1  # EPISODE_RESET and settlements just pass through
        return None

    # -------------------------------------------------------------- handlers

    def _execute_auction(self, event: MarketEvent, action: AuctionAction) -> list[ExecutionReport]:
        market = event.market
        assert market in AUCTION_EVENTS
        session = getattr(self.cfg.markets, market)
        reports = execute_auction_orders(
            market=market,
            session=session,
            event_products=event.products,
            orders=action.orders,
            clearing_prices=self._auction_prices[market],
            gate_utc=event.time_utc,
            book=self.book,
        )
        self._book_execution_cash(market, event.time_utc, reports)
        return reports

    def _execute_idc(self, event: MarketEvent, action: IdcAction) -> list[ExecutionReport]:
        reports = execute_idc_orders(
            cfg=self.cfg.markets.idc,
            event_products=event.products,
            orders=action.orders,
            decision_utc=event.time_utc,
            idc_indices=self._idc_indices,
            book=self.book,
        )
        self._book_execution_cash("idc", event.time_utc, reports)
        return reports

    def _book_execution_cash(
        self, market: str, time_utc: pd.Timestamp, reports: list[ExecutionReport]
    ) -> None:
        for report in reports:
            for trade in report.trades:
                self.ledger.add(
                    time_utc,
                    market,
                    trade.cash_flow_eur,
                    trade.product.start_utc,
                    note=f"trade {trade.trade_id} {trade.side} {trade.volume_mw:.3f} MW",
                )
                if trade.transaction_cost_eur:
                    self.ledger.add(
                        time_utc,
                        "transaction_cost",
                        -trade.transaction_cost_eur,
                        trade.product.start_utc,
                        note=f"trade {trade.trade_id}",
                    )

    def _dispatch(self, event: MarketEvent, action: DispatchAction) -> DispatchRecord:
        product = event.products[0]
        try:
            row = self.profiles.loc[product.start_utc]
        except KeyError as exc:
            raise KeyError(f"no site profile for {product.id}") from exc
        wind_avail = float(row["wind_avail_mw"])
        pv_avail = float(row["pv_avail_mw"])

        dispatch = project_dispatch(
            bess_power_mw=action.bess_power_mw,
            wind_curtail_mw=action.wind_curtail_mw,
            pv_curtail_mw=action.pv_curtail_mw,
            wind_avail_mw=wind_avail,
            pv_avail_mw=pv_avail,
            bess_bounds_mw=self.battery.power_bounds(product.duration),
            grid=self.cfg.site.grid,
        )
        battery = self.battery.apply(dispatch.bess_power_mw, product.duration)
        if abs(battery.applied_power_mw - dispatch.bess_power_mw) > 1e-6:
            raise AssertionError(
                "battery clipped a projected-feasible power — bounds inconsistent: "
                f"{dispatch.bess_power_mw} -> {battery.applied_power_mw}"
            )
        record = DispatchRecord(
            product=product,
            wind_avail_mw=wind_avail,
            pv_avail_mw=pv_avail,
            dispatch=dispatch,
            battery=battery,
        )
        self.dispatch_records[product.start_utc] = record
        return record

    def _settle(self, product: DeliveryProduct, time_utc: pd.Timestamp) -> SettlementResult:
        record = self.dispatch_records.get(product.start_utc)
        if record is None:
            raise RuntimeError(f"settlement before dispatch for {product.id}")
        delivered = energy_mwh(record.dispatch.grid_power_mw, product.duration)
        contracted = self.book.contracted_energy_mwh(product)
        settlement = self.settlement.settle(product, delivered, contracted)
        self.settlements[product.start_utc] = settlement

        self.ledger.add(
            time_utc,
            "imbalance",
            settlement.cash_eur,
            product.start_utc,
            note=f"deviation {settlement.deviation_mwh:+.4f} MWh @ "
            f"{settlement.settlement_price_eur_per_mwh:.2f}",
        )
        if settlement.penalty_eur:
            self.ledger.add(
                time_utc,
                "constraint_penalty",
                settlement.penalty_eur,
                product.start_utc,
                note="deviation penalty",
            )
        degradation = (
            -self.cfg.site.battery.degradation_cost_eur_per_mwh_throughput
            * record.battery.throughput_mwh
        )
        if degradation:
            self.ledger.add(time_utc, "degradation", degradation, product.start_utc)
        curtailed_mwh = energy_mwh(
            record.dispatch.wind_curtail_mw + record.dispatch.pv_curtail_mw, product.duration
        )
        if curtailed_mwh and self.cfg.site.curtailment_cost_eur_per_mwh:
            self.ledger.add(
                time_utc,
                "curtailment_penalty",
                -self.cfg.site.curtailment_cost_eur_per_mwh * curtailed_mwh,
                product.start_utc,
            )
        return settlement
