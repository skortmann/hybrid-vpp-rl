"""Action layouts: fixed-size action tensor <-> event-specific logical actions.

Several schema-versioned action formulations are supported (selected via
``episode.action_mode``); all of them translate into the same validated
logical actions (auction orders, IDC orders, dispatch requests), so the
simulator, accounting, and market rules are identical across variants.

``direct`` (act-v1)
    Box[-1,1]^(n_slots+3). One market slot per canonical quarter-hour of the
    episode window; slot value scales to a signed incremental MW order via
    the session's ``max_volume_mw``. Dispatch entries: BESS power and
    wind/PV curtailment. Inactive entries are masked and have no effect.

``target_position`` (act-v2)
    Same tensor shape, but a market slot expresses a **desired cumulative
    net position** in MW for that delivery quarter-hour (scaled to
    [-max_volume, +max_volume]); the translator submits the *difference* to
    the current position as the incremental order. The append-only ledger
    is preserved — only the request semantics change.

``hourly_target`` (act-v3)
    Low-dimensional variant of ``target_position``: one anchor per local
    hour of the episode window (24 per day) broadcast to its quarter-hours,
    plus the 3 dispatch entries. 27 dimensions for one-day episodes.

``residual_hourly`` (act-v4)
    ``hourly_target``-shaped tensor interpreted as a bounded correction
    around the rule-based controller's action: the baseline logical action
    is computed first (from the same non-privileged observations) and the
    policy adds up to ±``residual_scale_mw`` per hour anchor (markets) and
    a bounded correction to the baseline dispatch. Zero action reproduces
    the baseline exactly.
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

#: worst case quarter-hours / hours per local day (25-hour autumn day)
MAX_SLOTS_PER_DAY = 100
MAX_HOURS_PER_DAY = 25
N_DISPATCH_ENTRIES = 3

ACTION_SCHEMA_VERSIONS = {
    "direct": "act-v1-direct",
    "target_position": "act-v2-target",
    "hourly_target": "act-v3-hourly-target",
    "residual_hourly": "act-v4-residual-hourly",
}


@dataclass
class ActionLayout:
    cfg: ExperimentConfig
    episode_days: int = 1

    @property
    def mode(self) -> str:
        return self.cfg.episode.action_mode

    @property
    def schema_version(self) -> str:
        return ACTION_SCHEMA_VERSIONS[self.mode]

    @property
    def hourly(self) -> bool:
        return self.mode in ("hourly_target", "residual_hourly")

    @property
    def n_slots(self) -> int:
        per_day = MAX_HOURS_PER_DAY if self.hourly else MAX_SLOTS_PER_DAY
        return per_day * self.episode_days

    @property
    def size(self) -> int:
        return self.n_slots + N_DISPATCH_ENTRIES

    @property
    def needs_baseline(self) -> bool:
        return self.mode == "residual_hourly"

    def slot_of(self, window_start_utc: pd.Timestamp, product: DeliveryProduct) -> int:
        delta = product.start_utc - window_start_utc
        slot = int(delta / pd.Timedelta(hours=1)) if self.hourly else int(delta / INTERVAL)
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

    # ------------------------------------------------------------- translate

    def translate(
        self,
        raw: np.ndarray,
        window_start_utc: pd.Timestamp,
        event: MarketEvent,
        sim: Simulator,
        baseline: Action | None = None,
    ) -> Action:
        """Map a raw [-1, 1] vector to the logical action for this event.

        ``baseline`` is the rule-based logical action for this event and is
        required (and only used) in ``residual_hourly`` mode.
        """
        raw = np.asarray(raw, dtype=np.float64).clip(-1.0, 1.0)
        if raw.shape != (self.size,):
            raise ValueError(f"action shape {raw.shape}, expected {(self.size,)}")

        if self.mode == "residual_hourly":
            if baseline is None:
                raise ValueError("residual_hourly mode requires the baseline action")
            return self._translate_residual(raw, window_start_utc, event, sim, baseline)

        if event.type == EventType.PHYSICAL_DISPATCH:
            return self._dispatch_action(raw[self.n_slots :], event, sim)

        if event.type == EventType.IDC_DECISION:
            scale = self.cfg.markets.idc.max_volume_mw_per_trade
            orders: dict[DeliveryProduct, float] = {}
            for p in event.products:
                value = raw[self.slot_of(window_start_utc, p)]
                mw = self._order_mw(value, scale, sim, p)
                if abs(mw) > 1e-3 * scale:
                    orders[p] = float(mw)
            return IdcAction(orders)

        # auction gates
        session = getattr(self.cfg.markets, event.market)
        scale = session.max_volume_mw
        orders = {}
        for p in event.products:
            slots = sorted({self.slot_of(window_start_utc, qh) for qh in p.quarter_hours()})
            value = float(np.mean(raw[slots]))
            if self.mode == "direct":
                mw = value * scale
            else:  # target_position / hourly_target: desired cumulative position
                target = value * scale
                mw = target - sim.book.net_position_mw(p)
            if abs(mw) > 1e-3 * scale:
                orders[p] = float(mw)
        return AuctionAction(orders)

    def _order_mw(
        self, value: float, scale: float, sim: Simulator, product: DeliveryProduct
    ) -> float:
        if self.mode == "direct":
            return value * scale
        # target-position semantics for IDC as well
        target = value * scale
        return target - sim.book.net_position_mw(product)

    def _dispatch_action(self, entries: np.ndarray, event: MarketEvent, sim: Simulator):
        bat = self.cfg.site.battery
        b_raw, cw_raw, cpv_raw = entries
        bess = b_raw * bat.discharge_power_mw if b_raw >= 0 else b_raw * bat.charge_power_mw
        row = sim.profiles.loc[event.products[0].start_utc]
        # curtailment entries are mapped from [-1,1] to [0, available]
        return DispatchAction(
            bess_power_mw=float(bess),
            wind_curtail_mw=float((cw_raw + 1) / 2 * row["wind_avail_mw"]),
            pv_curtail_mw=float((cpv_raw + 1) / 2 * row["pv_avail_mw"]),
        )

    # -------------------------------------------------------------- residual

    def _translate_residual(
        self,
        raw: np.ndarray,
        window_start_utc: pd.Timestamp,
        event: MarketEvent,
        sim: Simulator,
        baseline: Action,
    ) -> Action:
        scale = self.cfg.episode.residual_scale_mw

        if event.type == EventType.PHYSICAL_DISPATCH:
            assert isinstance(baseline, DispatchAction)
            bat = self.cfg.site.battery
            b_raw, cw_raw, cpv_raw = raw[self.n_slots :]
            row = sim.profiles.loc[event.products[0].start_utc]
            power_span = max(bat.charge_power_mw, bat.discharge_power_mw)
            return DispatchAction(
                bess_power_mw=baseline.bess_power_mw + float(b_raw) * power_span,
                wind_curtail_mw=baseline.wind_curtail_mw
                + float(cw_raw) * float(row["wind_avail_mw"]),
                pv_curtail_mw=baseline.pv_curtail_mw + float(cpv_raw) * float(row["pv_avail_mw"]),
            )

        assert isinstance(baseline, AuctionAction | IdcAction)
        orders: dict[DeliveryProduct, float] = dict(baseline.orders)
        for p in event.products:
            slots = sorted({self.slot_of(window_start_utc, qh) for qh in p.quarter_hours()})
            correction = float(np.mean(raw[slots])) * scale
            if abs(correction) > 1e-3 * scale:
                orders[p] = orders.get(p, 0.0) + correction
        if event.type == EventType.IDC_DECISION:
            return IdcAction(orders)
        return AuctionAction(orders)
