"""Deviation (imbalance) settlement of delivered vs. contracted energy.

Sign conventions: deviation = delivered - contracted (MWh, positive = the
portfolio delivered more than it sold). Under the German single-price reBAP
scheme the balancing group's cash flow is simply::

    cash = rebap * deviation

(a long portfolio receives the — possibly negative — reBAP; a short
portfolio pays it). ``model="rebap"`` uses the actual historical reBAP and
is therefore not an approximation; the alternative models are clearly
stylized and reference the day-ahead price of the interval instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from hybrid_vpp.config.models import ImbalanceConfig
from hybrid_vpp.core.timegrid import DeliveryProduct


@dataclass(frozen=True, slots=True)
class SettlementResult:
    product: DeliveryProduct
    delivered_mwh: float
    contracted_mwh: float
    deviation_mwh: float
    settlement_price_eur_per_mwh: float
    cash_eur: float
    penalty_eur: float  # additional non-price penalty (<= 0)


class ImbalanceSettlement:
    def __init__(
        self,
        cfg: ImbalanceConfig,
        rebap: pd.Series,
        reference_prices: pd.Series | None = None,
    ) -> None:
        """``rebap``: 15-min UTC-indexed reBAP series.

        ``reference_prices``: quarter-hour UTC-indexed day-ahead reference
        used by the stylized models (required for those models only).
        """
        self.cfg = cfg
        self.rebap = rebap
        self.reference_prices = reference_prices
        if cfg.model != "rebap" and reference_prices is None:
            raise ValueError(f"imbalance model {cfg.model!r} needs reference_prices")

    def settle(
        self, product: DeliveryProduct, delivered_mwh: float, contracted_mwh: float
    ) -> SettlementResult:
        deviation = delivered_mwh - contracted_mwh
        price = self._price(product, deviation)
        cash = price * deviation
        penalty = -self.cfg.deviation_penalty_eur_per_mwh * abs(deviation)
        return SettlementResult(
            product=product,
            delivered_mwh=delivered_mwh,
            contracted_mwh=contracted_mwh,
            deviation_mwh=deviation,
            settlement_price_eur_per_mwh=price,
            cash_eur=cash,
            penalty_eur=penalty,
        )

    def _price(self, product: DeliveryProduct, deviation_mwh: float) -> float:
        if self.cfg.model == "rebap":
            price = self.rebap.get(product.start_utc)
            if price is None or pd.isna(price):
                raise KeyError(f"no reBAP for {product.id}")
            return float(price)

        ref = self.reference_prices.get(product.start_utc)
        if ref is None or pd.isna(ref):
            raise KeyError(f"no reference price for {product.id}")
        ref = float(ref)
        if self.cfg.model == "symmetric_penalty":
            # deviations valued at the DA reference; the fixed penalty is
            # applied via `deviation_penalty_eur_per_mwh` semantics below
            return ref - self.cfg.penalty_eur_per_mwh * _sign(deviation_mwh)
        if self.cfg.model == "asymmetric_spread":
            # long deviations sell below, short deviations buy above the reference
            return ref - self.cfg.spread_eur_per_mwh * _sign(deviation_mwh)
        raise ValueError(f"unknown imbalance model {self.cfg.model!r}")


def _sign(x: float) -> float:
    return 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)
