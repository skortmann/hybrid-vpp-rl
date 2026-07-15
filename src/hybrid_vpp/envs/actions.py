"""Fixed-size action tensor <-> event-specific logical actions.

The RL action is a Box[-1, 1] vector with a stable layout:

    [ market slots 0 .. n_slots-1 | bess | wind_curtail | pv_curtail ]

One market slot corresponds to one canonical quarter-hour of the episode's
delivery window (slot i = i-th 15-min interval after the window start; DST
days use fewer/more real slots, the rest stay masked). At each event only
the slots of *eligible* products are active — inactive entries are ignored
by construction and reported in the action mask. Market slots scale to
signed MW via the session's ``max_volume_mw``; dispatch entries scale to
the asset limits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from hybrid_vpp.config.models import ExperimentConfig
from hybrid_vpp.core.timegrid import INTERVAL, DeliveryProduct
from hybrid_vpp.markets.calendar import EventType, MarketEvent
from hybrid_vpp.sim.simulator import (
    Action,
    AuctionAction,
    DispatchAction,
    IdcAction,
    Simulator,
)

#: worst case quarter-hours per local day (25-hour autumn day)
MAX_SLOTS_PER_DAY = 100
N_DISPATCH_ENTRIES = 3


@dataclass
class ActionLayout:
    cfg: ExperimentConfig
    episode_days: int = 1

    @property
    def n_slots(self) -> int:
        return MAX_SLOTS_PER_DAY * self.episode_days

    @property
    def size(self) -> int:
        return self.n_slots + N_DISPATCH_ENTRIES

    def slot_of(self, window_start_utc: pd.Timestamp, product: DeliveryProduct) -> int:
        slot = int((product.start_utc - window_start_utc) / INTERVAL)
        if not 0 <= slot < self.n_slots:
            raise ValueError(f"product {product.id} outside episode window")
        return slot

    def mask(self, window_start_utc: pd.Timestamp, event: MarketEvent) -> np.ndarray:
        """1.0 for action entries that have an effect at this event."""
        mask = np.zeros(self.size, dtype=np.float32)
        if event.type == EventType.PHYSICAL_DISPATCH:
            mask[self.n_slots :] = 1.0
        else:
            for p in event.products:
                for qh in p.quarter_hours():
                    mask[self.slot_of(window_start_utc, qh)] = 1.0
        return mask

    def translate(
        self,
        raw: np.ndarray,
        window_start_utc: pd.Timestamp,
        event: MarketEvent,
        sim: Simulator,
    ) -> Action:
        """Map a raw [-1, 1] vector to the logical action for this event."""
        raw = np.asarray(raw, dtype=np.float64).clip(-1.0, 1.0)
        if raw.shape != (self.size,):
            raise ValueError(f"action shape {raw.shape}, expected {(self.size,)}")

        if event.type == EventType.PHYSICAL_DISPATCH:
            product = event.products[0]
            bat = self.cfg.site.battery
            b_raw, cw_raw, cpv_raw = raw[self.n_slots :]
            bess = b_raw * bat.discharge_power_mw if b_raw >= 0 else b_raw * bat.charge_power_mw
            row = sim.profiles.loc[product.start_utc]
            # curtailment entries are mapped from [-1,1] to [0, available]
            return DispatchAction(
                bess_power_mw=float(bess),
                wind_curtail_mw=float((cw_raw + 1) / 2 * row["wind_avail_mw"]),
                pv_curtail_mw=float((cpv_raw + 1) / 2 * row["pv_avail_mw"]),
            )

        if event.type == EventType.IDC_DECISION:
            scale = self.cfg.markets.idc.max_volume_mw_per_trade
            orders: dict[DeliveryProduct, float] = {}
            for p in event.products:
                value = raw[self.slot_of(window_start_utc, p)] * scale
                if abs(value) > 1e-3 * scale:
                    orders[p] = float(value)
            return IdcAction(orders)

        # auction gates: per-product volume = mean of its quarter-hour slots
        session = getattr(self.cfg.markets, event.market)
        orders = {}
        for p in event.products:
            slots = [self.slot_of(window_start_utc, qh) for qh in p.quarter_hours()]
            value = float(np.mean(raw[slots])) * session.max_volume_mw
            if abs(value) > 1e-3 * session.max_volume_mw:
                orders[p] = value
        return AuctionAction(orders)
