"""Strategic action translator (act-v5): 7 economic decision variables.

Instead of per-product orders, the policy outputs low-dimensional strategic
quantities; a deterministic translator maps them onto feasible logical
actions, reusing the rule-based controller as the structural backbone (its
forecast plumbing and daily battery-arbitrage plan). A mid-range action
(coverage 1, gains 1, threshold 0, bias 0) reproduces rule-based behavior.

| dim | name | event | range | meaning |
|-----|------|-------|-------|---------|
| 0 | daa_coverage | DAA | [0, 1.2] | sold share of the renewable forecast |
| 1 | arbitrage_scale | DAA | [0, 1] | scale of the battery arbitrage block |
| 2 | ida_gain | IDA1-3 | [0, 1] | fraction of the rule-based correction traded |
| 3 | idc_gain | IDC | [0, 1] | fraction of the rule-based correction traded |
| 4 | bess_tracking_gain | dispatch | [0, 1] | position-tracking intensity of the BESS |
| 5 | curtail_threshold | dispatch | [-100, 100] €/MWh | curtail surplus below this price forecast |
| 6 | soc_bias | dispatch | [-1, 1] | steady charge(-)/discharge(+) preference |

All raw entries live in [-1, 1] and are rescaled here. Only the dims listed
for the current event are active (masked otherwise).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
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

N_STRATEGIC_DIMS = 7

#: active action dims per event type
STRATEGIC_MASKS: dict[EventType, tuple[int, ...]] = {
    EventType.DAA_GATE_CLOSURE: (0, 1),
    EventType.IDA1_GATE_CLOSURE: (2,),
    EventType.IDA2_GATE_CLOSURE: (2,),
    EventType.IDA3_GATE_CLOSURE: (2,),
    EventType.IDC_DECISION: (3,),
    EventType.PHYSICAL_DISPATCH: (4, 5, 6),
}


def _unit(a: float) -> float:
    """[-1,1] -> [0,1]."""
    return float(np.clip((a + 1.0) / 2.0, 0.0, 1.0))


@dataclass
class StrategicTranslator:
    """Deterministic map from strategic parameters to logical actions."""

    cfg: ExperimentConfig
    rule_based: object  # RuleBasedController (provides plans + forecasters)
    price_forecaster: object

    def translate(self, raw: np.ndarray, event: MarketEvent, sim: Simulator) -> Action:
        baseline = self.rule_based.act(event, sim)

        if event.type == EventType.DAA_GATE_CLOSURE:
            coverage = _unit(raw[0]) * 1.2
            arb_scale = _unit(raw[1])
            assert isinstance(baseline, AuctionAction)
            qh_times = pd.DatetimeIndex(
                [qh.start_utc for p in event.products for qh in p.quarter_hours()]
            )
            fc = self.rule_based.renewable_forecaster.forecast(event.time_utc, qh_times)
            renewable = (fc["wind_mw"] + fc["pv_mw"]).clip(upper=self.cfg.site.grid.export_limit_mw)
            orders = {}
            for p in event.products:
                qhs = [qh.start_utc for qh in p.quarter_hours()]
                fc_mw = float(renewable.reindex(qhs).mean())
                rb_mw = baseline.orders.get(p, 0.0)
                plan_mw = rb_mw - fc_mw  # rule-based battery-arbitrage component
                mw = coverage * fc_mw + arb_scale * plan_mw
                if abs(mw) > 1e-3:
                    orders[p] = mw
            return AuctionAction(orders)

        gain_max = self.cfg.episode.strategic_gain_max

        if event.type in (
            EventType.IDA1_GATE_CLOSURE,
            EventType.IDA2_GATE_CLOSURE,
            EventType.IDA3_GATE_CLOSURE,
        ):
            gain = _unit(raw[2]) * gain_max
            assert isinstance(baseline, AuctionAction)
            return AuctionAction(
                {p: gain * mw for p, mw in baseline.orders.items() if abs(gain * mw) > 1e-3}
            )

        if event.type == EventType.IDC_DECISION:
            gain = _unit(raw[3]) * gain_max
            assert isinstance(baseline, IdcAction)
            return IdcAction(
                {p: gain * mw for p, mw in baseline.orders.items() if abs(gain * mw) > 1e-3}
            )

        # physical dispatch
        assert isinstance(baseline, DispatchAction)
        if self.cfg.episode.strategic_fixed_dispatch:
            return baseline  # hybrid H4: deterministic rule-based dispatch
        tracking = min(_unit(raw[4]) * gain_max, 1.0)
        threshold = float(raw[5]) * 100.0
        bias = float(raw[6])
        bat = self.cfg.site.battery
        product = event.products[0]
        row = sim.profiles.loc[product.start_utc]
        renewables = float(row["wind_avail_mw"] + row["pv_avail_mw"])
        position = sim.book.net_position_mw(product)

        bess = tracking * (position - renewables) + bias * bat.discharge_power_mw / 2.0

        price = self.price_forecaster.forecast(
            "daa", event.time_utc, pd.DatetimeIndex([product.start_utc])
        )
        curtail = 0.0
        if float(price.iloc[0]) < threshold:
            p_min, _ = sim.battery.power_bounds(product.duration)
            surplus = max(0.0, renewables - position - (-p_min))
            curtail = surplus
        share = float(row["wind_avail_mw"]) / renewables if renewables > 0 else 0.0
        return DispatchAction(
            bess_power_mw=bess,
            wind_curtail_mw=curtail * share,
            pv_curtail_mw=curtail * (1.0 - share),
        )
