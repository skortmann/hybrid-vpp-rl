"""Trades, position book, and cash-flow ledger.

Conventions
-----------
* A :class:`Trade` is an immutable, append-only fill on one canonical
  quarter-hour delivery interval. Hourly DAA products are recorded as four
  quarter-hour fills sharing a ``parent_product`` (constant MW across the
  hour, one price) so that delivery accounting has a single granularity.
* Position sign: **positive net position = net sold** = committed export.
* Cash-flow sign: sells are positive (revenue), buys negative.
* Market adjustments are additive transactions; earlier positions are never
  overwritten. The book rejects trades executed at or after delivery start —
  market-specific gate rules are enforced by the execution layer on top.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from hybrid_vpp.core.timegrid import DeliveryProduct

Side = Literal["buy", "sell"]

MARKETS = ("daa", "ida1", "ida2", "ida3", "idc")


@dataclass(frozen=True, slots=True)
class Trade:
    trade_id: int
    market: str
    product: DeliveryProduct  # canonical quarter-hour interval
    side: Side
    volume_mw: float  # > 0
    price_eur_per_mwh: float
    executed_utc: pd.Timestamp
    transaction_cost_eur: float = 0.0
    parent_product: DeliveryProduct | None = None

    @property
    def energy_mwh(self) -> float:
        return self.volume_mw * self.product.hours

    @property
    def signed_energy_mwh(self) -> float:
        """Positive = sold energy."""
        return self.energy_mwh if self.side == "sell" else -self.energy_mwh

    @property
    def cash_flow_eur(self) -> float:
        """Revenue from sells, cost of buys (transaction cost excluded)."""
        return self.signed_energy_mwh * self.price_eur_per_mwh


class PositionBook:
    """Append-only record of all fills, queryable per delivery interval."""

    def __init__(self) -> None:
        self._trades: list[Trade] = []
        self._by_product: dict[pd.Timestamp, list[Trade]] = defaultdict(list)
        self._ids = itertools.count()

    def next_trade_id(self) -> int:
        return next(self._ids)

    @property
    def trades(self) -> tuple[Trade, ...]:
        return tuple(self._trades)

    def add(self, trade: Trade) -> None:
        if trade.volume_mw <= 0:
            raise ValueError(f"trade volume must be positive, got {trade.volume_mw}")
        if trade.executed_utc >= trade.product.start_utc:
            raise ValueError(
                f"retroactive trade rejected: executed {trade.executed_utc} "
                f">= delivery start {trade.product.start_utc}"
            )
        self._trades.append(trade)
        self._by_product[trade.product.start_utc].append(trade)

    # ------------------------------------------------------------- queries

    def trades_for(self, product: DeliveryProduct) -> list[Trade]:
        return list(self._by_product.get(product.start_utc, ()))

    def net_position_mw(self, product: DeliveryProduct, market: str | None = None) -> float:
        """Net sold MW for one quarter-hour (positive = committed export)."""
        return self.net_position_mw_at(product.start_utc, market)

    def net_position_mw_at(self, start_utc: pd.Timestamp, market: str | None = None) -> float:
        trades = self._by_product.get(start_utc, ())
        return sum(
            (t.volume_mw if t.side == "sell" else -t.volume_mw)
            for t in trades
            if market is None or t.market == market
        )

    def contracted_energy_mwh(self, product: DeliveryProduct) -> float:
        return sum(t.signed_energy_mwh for t in self._by_product.get(product.start_utc, ()))

    def market_breakdown_mw(self, product: DeliveryProduct) -> dict[str, float]:
        return {m: self.net_position_mw(product, m) for m in MARKETS}

    def turnover_mwh(self, market: str | None = None) -> float:
        return sum(t.energy_mwh for t in self._trades if market is None or t.market == market)

    def to_frame(self) -> pd.DataFrame:
        rows = [
            {
                "trade_id": t.trade_id,
                "market": t.market,
                "delivery_start_utc": t.product.start_utc,
                "side": t.side,
                "volume_mw": t.volume_mw,
                "energy_mwh": t.energy_mwh,
                "price_eur_per_mwh": t.price_eur_per_mwh,
                "cash_flow_eur": t.cash_flow_eur,
                "transaction_cost_eur": t.transaction_cost_eur,
                "executed_utc": t.executed_utc,
            }
            for t in self._trades
        ]
        return pd.DataFrame(rows)


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    time_utc: pd.Timestamp
    component: str  # daa | ida1 | ida2 | ida3 | idc | imbalance | transaction_cost | ...
    amount_eur: float
    delivery_start_utc: pd.Timestamp | None = None
    note: str = ""


class Ledger:
    """Append-only cash-flow ledger; every euro is booked exactly once."""

    COMPONENTS = (
        *MARKETS,
        "imbalance",
        "transaction_cost",
        "degradation",
        "curtailment_penalty",
        "constraint_penalty",
    )

    def __init__(self) -> None:
        self._entries: list[LedgerEntry] = []

    def add(
        self,
        time_utc: pd.Timestamp,
        component: str,
        amount_eur: float,
        delivery_start_utc: pd.Timestamp | None = None,
        note: str = "",
    ) -> None:
        if component not in self.COMPONENTS:
            raise ValueError(f"unknown ledger component {component!r}")
        self._entries.append(
            LedgerEntry(time_utc, component, float(amount_eur), delivery_start_utc, note)
        )

    @property
    def entries(self) -> tuple[LedgerEntry, ...]:
        return tuple(self._entries)

    def total(self, component: str | None = None) -> float:
        return sum(e.amount_eur for e in self._entries if component in (None, e.component))

    def by_component(self) -> dict[str, float]:
        out: dict[str, float] = dict.fromkeys(self.COMPONENTS, 0.0)
        for e in self._entries:
            out[e.component] += e.amount_eur
        return out

    def entries_since(self, index: int) -> tuple[LedgerEntry, ...]:
        return tuple(self._entries[index:])

    def __len__(self) -> int:
        return len(self._entries)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([e.__dict__ for e in self._entries])
