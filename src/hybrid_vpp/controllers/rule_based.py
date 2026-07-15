"""Rule-based baseline controller.

Strategy (each rule is deliberately simple and fully documented):

1. **DAA gate** — sell the renewable forecast per product (capped at the
   grid export limit), but offer nothing for products whose price forecast
   is below ``min_sell_price``. Optionally overlay a day-ahead battery
   arbitrage block: buy ``charge_power`` in the ``k`` cheapest forecast
   hours, sell ``discharge_power`` in the ``k`` most expensive hours, with
   ``k`` derived from the usable battery energy.
2. **IDA gates** — re-forecast and trade the difference between the target
   position (updated forecast + battery plan) and the current contracted
   position, if it exceeds ``ida_threshold_mw``.
3. **IDC decisions** — same correction for products starting within
   ``idc_horizon``, threshold ``idc_threshold_mw``.
4. **Physical dispatch** — request BESS power equal to the gap between the
   contracted position and available renewables (discharge when short,
   charge when long); request curtailment for surplus that the battery
   cannot absorb when the interval's price forecast is negative. The
   feasibility layer enforces the grid limit on top.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from hybrid_vpp.config.models import ExperimentConfig
from hybrid_vpp.core.timegrid import DeliveryProduct
from hybrid_vpp.markets.calendar import EventType, MarketEvent
from hybrid_vpp.sim.simulator import (
    Action,
    AuctionAction,
    DispatchAction,
    IdcAction,
    Simulator,
)


@dataclass
class RuleBasedController:
    cfg: ExperimentConfig
    renewable_forecaster: object  # RenewableForecastProvider
    price_forecaster: object  # PriceForecastProvider
    use_battery_arbitrage: bool = True
    min_sell_price: float = 0.0  # EUR/MWh — below this, don't offer DA volume
    ida_threshold_mw: float = 1.0
    idc_threshold_mw: float = 0.5
    idc_horizon: pd.Timedelta = pd.Timedelta("4h")
    #: planned battery MW per quarter-hour start (positive = discharge)
    _bess_plan: dict[pd.Timestamp, float] = field(default_factory=dict)

    def reset(self) -> None:
        self._bess_plan = {}

    # ----------------------------------------------------------------- events

    def act(self, event: MarketEvent, sim: Simulator) -> Action:
        if event.type == EventType.DAA_GATE_CLOSURE:
            return self._act_daa(event, sim)
        if event.type in (
            EventType.IDA1_GATE_CLOSURE,
            EventType.IDA2_GATE_CLOSURE,
            EventType.IDA3_GATE_CLOSURE,
        ):
            return self._act_ida(event, sim)
        if event.type == EventType.IDC_DECISION:
            return self._act_idc(event, sim)
        if event.type == EventType.PHYSICAL_DISPATCH:
            return self._act_dispatch(event, sim)
        raise ValueError(f"unexpected event {event.type}")

    def _act_daa(self, event: MarketEvent, sim: Simulator) -> AuctionAction:
        products = event.products
        qh_times = pd.DatetimeIndex([qh.start_utc for p in products for qh in p.quarter_hours()])
        fc = self.renewable_forecaster.forecast(event.time_utc, qh_times)
        renewable_qh = (fc["wind_mw"] + fc["pv_mw"]).clip(upper=self.cfg.site.grid.export_limit_mw)
        price_fc = self.price_forecaster.forecast(
            "daa", event.time_utc, pd.DatetimeIndex([p.start_utc for p in products])
        )

        if self.use_battery_arbitrage:
            self._plan_battery(products, price_fc)

        orders: dict[DeliveryProduct, float] = {}
        for p in products:
            qhs = [qh.start_utc for qh in p.quarter_hours()]
            volume = float(renewable_qh.reindex(qhs).mean())
            plan = float(np.mean([self._bess_plan.get(t, 0.0) for t in qhs]))
            if price_fc.get(p.start_utc, 0.0) < self.min_sell_price:
                volume = 0.0  # expected negative price: don't sell the forecast
            orders[p] = volume + plan
        return AuctionAction(orders)

    def _plan_battery(self, products: tuple[DeliveryProduct, ...], price_fc: pd.Series) -> None:
        """One charge/discharge block per day, sized by usable energy."""
        bat = self.cfg.site.battery
        usable_mwh = (bat.soc_max - bat.soc_min) * bat.energy_capacity_mwh
        prices = pd.Series({p.start_utc: price_fc.get(p.start_utc, 0.0) for p in products})
        product_hours = products[0].hours
        charge_hours = usable_mwh / (bat.charge_power_mw * bat.charge_efficiency)
        discharge_hours = usable_mwh * bat.discharge_efficiency / bat.discharge_power_mw
        # rank product prices: cheapest -> charge, most expensive -> discharge
        cheap = prices.nsmallest(max(1, round(charge_hours / product_hours))).index
        rich = prices.nlargest(max(1, round(discharge_hours / product_hours))).index
        overlap = set(cheap) & set(rich)
        cheap = [t for t in cheap if t not in overlap]
        rich = [t for t in rich if t not in overlap]
        for p in products:
            power = 0.0
            if p.start_utc in cheap:
                power = -bat.charge_power_mw
            elif p.start_utc in rich:
                power = bat.discharge_power_mw
            for qh in p.quarter_hours():
                self._bess_plan[qh.start_utc] = power

    def _target_position_mw(
        self, sim: Simulator, event: MarketEvent, products: tuple[DeliveryProduct, ...]
    ) -> pd.Series:
        times = pd.DatetimeIndex([p.start_utc for p in products])
        fc = self.renewable_forecaster.forecast(event.time_utc, times)
        renewable = (fc["wind_mw"] + fc["pv_mw"]).clip(upper=self.cfg.site.grid.export_limit_mw)
        plan = pd.Series([self._bess_plan.get(t, 0.0) for t in times], index=times)
        return renewable + plan

    def _act_ida(self, event: MarketEvent, sim: Simulator) -> AuctionAction:
        target = self._target_position_mw(sim, event, event.products)
        orders = {}
        for p in event.products:
            delta = float(target[p.start_utc]) - sim.book.net_position_mw(p)
            if abs(delta) >= self.ida_threshold_mw:
                orders[p] = delta
        return AuctionAction(orders)

    def _act_idc(self, event: MarketEvent, sim: Simulator) -> IdcAction:
        near = tuple(p for p in event.products if p.start_utc - event.time_utc <= self.idc_horizon)
        if not near:
            return IdcAction()
        target = self._target_position_mw(sim, event, near)
        orders = {}
        for p in near:
            delta = float(target[p.start_utc]) - sim.book.net_position_mw(p)
            if abs(delta) >= self.idc_threshold_mw:
                orders[p] = delta
        return IdcAction(orders)

    def _act_dispatch(self, event: MarketEvent, sim: Simulator) -> DispatchAction:
        product = event.products[0]
        row = sim.profiles.loc[product.start_utc]
        renewables = float(row["wind_avail_mw"] + row["pv_avail_mw"])
        position = sim.book.net_position_mw(product)
        bess_request = position - renewables  # discharge when short, charge when long

        curtail = 0.0
        price_fc = self.price_forecaster.forecast(
            "daa", event.time_utc, pd.DatetimeIndex([product.start_utc])
        )
        if float(price_fc.iloc[0]) < 0.0:
            p_min, _ = sim.battery.power_bounds(product.duration)
            absorbable = -p_min
            curtail = max(0.0, renewables - position - absorbable)
        # split requested curtailment proportionally between wind and PV
        wind_share = row["wind_avail_mw"] / renewables if renewables > 0 else 0.0
        return DispatchAction(
            bess_power_mw=bess_request,
            wind_curtail_mw=curtail * wind_share,
            pv_curtail_mw=curtail * (1.0 - wind_share),
        )
