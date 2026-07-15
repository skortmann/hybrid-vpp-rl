"""Execution models: auction clearing and IDC price-taker fills.

Both models are **price-taker** approximations and say so loudly:

* Auctions (DAA, IDA1-3) fill the full requested volume at the historical
  clearing price of the product. Our volumes are assumed too small to move
  the auction clearing — no market impact.
* IDC fills execute at a historical VWAP index of the product (ID1 for
  remaining lead <= 1 h, ID3 for <= 3 h, IDFULL otherwise; configurable).
  No order book, no partial fills, no bid/ask spread (a spread model can be
  layered on via ``transaction_cost_eur_per_mwh``). See docs for the list
  of omitted real-market effects.

Every request is validated against the market gates; rejected requests are
returned with an explicit reason, never silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from hybrid_vpp.config.models import AuctionSessionConfig, IdcConfig
from hybrid_vpp.core.timegrid import DeliveryProduct
from hybrid_vpp.markets.positions import PositionBook, Trade

_TOL = 1e-9


@dataclass(frozen=True, slots=True)
class ExecutionReport:
    """Outcome of one requested order."""

    product: DeliveryProduct
    requested_mw: float  # signed: + sell, - buy
    filled_mw: float  # signed, 0 if rejected
    price_eur_per_mwh: float | None
    trades: tuple[Trade, ...]
    reason: str | None = None  # set when not (fully) filled as requested


def execute_auction_orders(
    *,
    market: str,
    session: AuctionSessionConfig,
    event_products: tuple[DeliveryProduct, ...],
    orders: dict[DeliveryProduct, float],
    clearing_prices: pd.Series,
    gate_utc: pd.Timestamp,
    book: PositionBook,
) -> list[ExecutionReport]:
    """Fill signed MW orders (+sell / -buy) at historical clearing prices.

    Orders for products outside the event's eligible set raise — the caller
    (simulator/env) must never let an agent trade outside its auction window.
    Products without a historical clearing price (cancelled auction days)
    are rejected with reason ``no_clearing_price``.
    """
    eligible = set(event_products)
    reports: list[ExecutionReport] = []
    for product, requested in orders.items():
        if product not in eligible:
            raise ValueError(
                f"{market}: order for {product.id} outside auction scope at gate {gate_utc}"
            )
        if abs(requested) < _TOL:
            continue
        capped = max(-session.max_volume_mw, min(session.max_volume_mw, requested))
        reason = "volume capped" if capped != requested else None

        price = clearing_prices.get(product.start_utc)
        if price is None or pd.isna(price):
            reports.append(ExecutionReport(product, requested, 0.0, None, (), "no_clearing_price"))
            continue

        side = "sell" if capped > 0 else "buy"
        volume = abs(capped)
        fills: list[Trade] = []
        # hourly (or longer) products fill as constant-MW quarter-hour trades
        for qh in product.quarter_hours():
            cost = session.transaction_cost_eur_per_mwh * volume * qh.hours
            trade = Trade(
                trade_id=book.next_trade_id(),
                market=market,
                product=qh,
                side=side,
                volume_mw=volume,
                price_eur_per_mwh=float(price),
                executed_utc=gate_utc,
                transaction_cost_eur=cost,
                parent_product=product if product.duration != qh.duration else None,
            )
            book.add(trade)
            fills.append(trade)
        reports.append(
            ExecutionReport(product, requested, capped, float(price), tuple(fills), reason)
        )
    return reports


def select_idc_index(lead: pd.Timedelta, cfg: IdcConfig) -> list[str]:
    """Preferred execution index for a remaining lead time, with fallbacks."""
    mapping = cfg.execution_index_by_lead
    ordered: list[tuple[pd.Timedelta, str]] = []
    for key, name in mapping.items():
        horizon = pd.Timedelta.max if key == "inf" else pd.Timedelta(key)
        ordered.append((horizon, name))
    ordered.sort(key=lambda kv: kv[0])
    preferred = [name for horizon, name in ordered if lead <= horizon]
    rest = [name for _, name in ordered if name not in preferred]
    return preferred + rest


def execute_idc_orders(
    *,
    cfg: IdcConfig,
    event_products: tuple[DeliveryProduct, ...],
    orders: dict[DeliveryProduct, float],
    decision_utc: pd.Timestamp,
    idc_indices: pd.DataFrame,
    book: PositionBook,
) -> list[ExecutionReport]:
    """Fill signed MW orders at the historical IDC index for the lead time."""
    eligible = set(event_products)
    reports: list[ExecutionReport] = []
    for product, requested in orders.items():
        if product not in eligible:
            raise ValueError(
                f"idc: order for {product.id} not tradable at {decision_utc} "
                "(gate closed or not yet open)"
            )
        if abs(requested) < _TOL:
            continue
        capped = max(-cfg.max_volume_mw_per_trade, min(cfg.max_volume_mw_per_trade, requested))
        reason = "volume capped" if capped != requested else None

        price = None
        if product.start_utc in idc_indices.index:
            row = idc_indices.loc[product.start_utc]
            lead = product.start_utc - decision_utc
            for name in select_idc_index(lead, cfg):
                candidate = row.get(name)
                if candidate is not None and pd.notna(candidate):
                    price = float(candidate)
                    break
        if price is None:
            reports.append(ExecutionReport(product, requested, 0.0, None, (), "no_price_data"))
            continue

        side = "sell" if capped > 0 else "buy"
        volume = abs(capped)
        cost = cfg.transaction_cost_eur_per_mwh * volume * product.hours
        trade = Trade(
            trade_id=book.next_trade_id(),
            market="idc",
            product=product,
            side=side,
            volume_mw=volume,
            price_eur_per_mwh=price,
            executed_utc=decision_utc,
            transaction_cost_eur=cost,
        )
        book.add(trade)
        reports.append(ExecutionReport(product, requested, capped, price, (trade,), reason))
    return reports
