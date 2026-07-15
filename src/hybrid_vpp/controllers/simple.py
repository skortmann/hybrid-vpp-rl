"""Must-take baseline: sell the day-ahead forecast, never touch the battery.

No intraday corrections, no strategic battery use, no voluntary
curtailment. Forecast errors settle at the imbalance price; the grid limit
is still enforced physically (technical curtailment may occur via the
feasibility layer when the battery cannot absorb excess).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from hybrid_vpp.config.models import ExperimentConfig
from hybrid_vpp.markets.calendar import EventType, MarketEvent
from hybrid_vpp.sim.simulator import (
    Action,
    AuctionAction,
    DispatchAction,
    IdcAction,
    Simulator,
)


@dataclass
class DoNothingController:
    cfg: ExperimentConfig
    renewable_forecaster: object

    def reset(self) -> None:
        pass

    def act(self, event: MarketEvent, sim: Simulator) -> Action:
        if event.type == EventType.DAA_GATE_CLOSURE:
            qh_times = pd.DatetimeIndex(
                [qh.start_utc for p in event.products for qh in p.quarter_hours()]
            )
            fc = self.renewable_forecaster.forecast(event.time_utc, qh_times)
            total = (fc["wind_mw"] + fc["pv_mw"]).clip(upper=self.cfg.site.grid.export_limit_mw)
            orders = {}
            for p in event.products:
                qhs = [qh.start_utc for qh in p.quarter_hours()]
                orders[p] = float(total.reindex(qhs).mean())
            return AuctionAction(orders)
        if event.type == EventType.PHYSICAL_DISPATCH:
            return DispatchAction()  # no battery, no voluntary curtailment
        if event.type == EventType.IDC_DECISION:
            return IdcAction()
        return AuctionAction()
