"""Rolling-horizon deterministic optimization benchmark.

At every auction gate the controller solves a MILP over the auction's
delivery horizon using exactly the forecasts available to the other
controllers, then trades the difference between the optimal export
schedule and the current contracted position. Physical dispatch follows
the contracted position (identical rule to the rule-based baseline), so
the benchmark differs from it only in *scheduling* quality.

MILP per solve (quarter-hour grid ``t``):

    max  sum_t h * [ price_t * g_t - c_deg * (ch_t + dis_t) ]
    s.t. g_t = wind_fc_t - cw_t + pv_fc_t - cpv_t + dis_t - ch_t
         0 <= ch_t <= P_ch * u_t          (u_t binary)
         0 <= dis_t <= P_dis * (1 - u_t)
         E_{t+1} = E_t + h * (eta_ch * ch_t - dis_t / eta_dis)
         E_min <= E_t <= E_max,  E_0 given,  E_T >= E_terminal
         0 <= cw_t <= wind_fc_t,  0 <= cpv_t <= pv_fc_t
         -P_import <= g_t <= P_export

The binaries forbid simultaneous charge/discharge, which the LP relaxation
would otherwise exploit at negative prices. With perfect-foresight
forecasters this same controller is the perfect-foresight upper bound.

The model is built once in PyOptInterface; the backend solver is
interchangeable (``gurobi`` by default, ``highs`` as the license-free
fallback — selected automatically when Gurobi is unavailable).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pyoptinterface as poi

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

log = logging.getLogger(__name__)

_RESOLVED_BACKEND: dict[str, type] = {}


def _make_model(solver: str):
    """Fresh PyOptInterface model for the requested backend (cached probe)."""
    factory = _RESOLVED_BACKEND.get(solver)
    if factory is None:
        if solver == "gurobi":
            try:
                from pyoptinterface import gurobi

                gurobi.Model()  # probes availability and license once
                factory = gurobi.Model
            except Exception:  # pragma: no cover - license-dependent
                log.warning("gurobi unavailable — falling back to HiGHS")
                from pyoptinterface import highs

                factory = highs.Model
        elif solver == "highs":
            from pyoptinterface import highs

            factory = highs.Model
        else:
            raise ValueError(f"unknown solver {solver!r} (use 'gurobi' or 'highs')")
        _RESOLVED_BACKEND[solver] = factory
    return factory()


@dataclass
class ScheduleSolution:
    export_mw: pd.Series  # planned grid power per quarter-hour start
    bess_mw: pd.Series  # planned signed BESS power
    objective_eur: float


@dataclass
class OptimizationController:
    cfg: ExperimentConfig
    renewable_forecaster: object
    price_forecaster: object
    #: also re-optimize position at IDC decisions for near products
    idc_corrections: bool = True
    idc_horizon: pd.Timedelta = pd.Timedelta("4h")
    idc_threshold_mw: float = 0.5
    #: PyOptInterface backend: "gurobi" (default, falls back) or "highs"
    solver: str = "gurobi"
    #: EUR/MWh penalty on schedule changes vs the previous own plan
    #: (benchmark variant; 0 = classic profit-maximizing MILP)
    turnover_penalty_eur_per_mwh: float = 0.0
    #: multiplicative derating of renewable forecasts (robust variant; 1 = trust)
    renewable_derate: float = 1.0
    _plan_export: dict[pd.Timestamp, float] = field(default_factory=dict)
    _plan_bess: dict[pd.Timestamp, float] = field(default_factory=dict)

    def reset(self) -> None:
        self._plan_export = {}
        self._plan_bess = {}

    # ------------------------------------------------------------------ MILP

    def solve_schedule(
        self,
        times: pd.DatetimeIndex,
        wind_fc: np.ndarray,
        pv_fc: np.ndarray,
        prices: np.ndarray,
        soc0_mwh: float,
        terminal_mwh: float | None,
        anchor_export_mw: np.ndarray | None = None,
    ) -> ScheduleSolution:
        """Build and solve the schedule MILP (backend per ``self.solver``)."""
        bat = self.cfg.site.battery
        grid = self.cfg.site.grid
        n = len(times)
        h = 0.25
        c_deg = bat.degradation_cost_eur_per_mwh_throughput
        renew = wind_fc + pv_fc
        import_limit = grid.import_limit_mw if grid.allow_grid_import else 0.0
        e_min = bat.soc_min * bat.energy_capacity_mwh
        e_max = bat.soc_max * bat.energy_capacity_mwh

        m = _make_model(self.solver)
        m.set_model_attribute(poi.ModelAttribute.Silent, True)
        m.set_model_attribute(poi.ModelAttribute.TimeLimitSec, 30.0)

        ch = [m.add_variable(lb=0.0, ub=bat.charge_power_mw) for _ in range(n)]
        dis = [m.add_variable(lb=0.0, ub=bat.discharge_power_mw) for _ in range(n)]
        cw = [m.add_variable(lb=0.0, ub=float(wind_fc[t])) for t in range(n)]
        cpv = [m.add_variable(lb=0.0, ub=float(pv_fc[t])) for t in range(n)]
        u = [m.add_variable(domain=poi.VariableDomain.Binary) for _ in range(n)]

        soc = 0.0 * ch[0] + soc0_mwh  # affine expression, grows term by term
        for t in range(n):
            m.add_linear_constraint(ch[t] - bat.charge_power_mw * u[t], poi.Leq, 0.0)
            m.add_linear_constraint(
                dis[t] + bat.discharge_power_mw * u[t], poi.Leq, bat.discharge_power_mw
            )
            g = dis[t] - ch[t] - cw[t] - cpv[t]
            m.add_linear_constraint(g, poi.Leq, grid.export_limit_mw - float(renew[t]))
            m.add_linear_constraint(g, poi.Geq, -import_limit - float(renew[t]))
            soc = (
                soc + (h * bat.charge_efficiency) * ch[t] - (h / bat.discharge_efficiency) * dis[t]
            )
            m.add_linear_constraint(soc, poi.Geq, e_min)
            m.add_linear_constraint(soc, poi.Leq, e_max)
        if terminal_mwh is not None:
            m.add_linear_constraint(soc, poi.Geq, float(terminal_mwh))

        objective = poi.ExprBuilder()
        for t in range(n):
            price_h = h * float(prices[t])
            objective += price_h * (dis[t] - ch[t] - cw[t] - cpv[t])
            objective -= (c_deg * h) * (ch[t] + dis[t])
        if self.turnover_penalty_eur_per_mwh > 0 and anchor_export_mw is not None:
            c_turn = self.turnover_penalty_eur_per_mwh * h
            for t in range(n):
                if np.isnan(anchor_export_mw[t]):
                    continue  # no previous plan for this quarter-hour
                slack = m.add_variable(lb=0.0)
                g = dis[t] - ch[t] - cw[t] - cpv[t] + float(renew[t])
                m.add_linear_constraint(slack - g, poi.Geq, -float(anchor_export_mw[t]))
                m.add_linear_constraint(slack + g, poi.Geq, float(anchor_export_mw[t]))
                objective -= c_turn * slack
        m.set_objective(objective, poi.ObjectiveSense.Maximize)
        m.optimize()

        status = m.get_model_attribute(poi.ModelAttribute.TerminationStatus)
        ok = (poi.TerminationStatusCode.OPTIMAL, poi.TerminationStatusCode.TIME_LIMIT)
        if status not in ok:
            raise RuntimeError(f"schedule MILP failed: {status}")

        bess = np.array([m.get_value(dis[t]) - m.get_value(ch[t]) for t in range(n)])
        curt = np.array([m.get_value(cw[t]) + m.get_value(cpv[t]) for t in range(n)])
        export = renew - curt + bess
        variable_part = float(m.get_model_attribute(poi.ModelAttribute.ObjectiveValue))
        return ScheduleSolution(
            export_mw=pd.Series(export, index=times),
            bess_mw=pd.Series(bess, index=times),
            objective_eur=variable_part + float(np.dot(prices, renew) * h),
        )

    # ---------------------------------------------------------------- events

    def _reoptimize(
        self, event: MarketEvent, sim: Simulator, market: str
    ) -> dict[pd.Timestamp, float]:
        """Solve over the event's products; return target export MW per QH."""
        qh_times = pd.DatetimeIndex(
            [qh.start_utc for p in event.products for qh in p.quarter_hours()]
        )
        fc = self.renewable_forecaster.forecast(event.time_utc, qh_times)
        price_qh = self.price_forecaster.forecast(market, event.time_utc, qh_times)
        bat = self.cfg.site.battery
        anchor = None
        if self.turnover_penalty_eur_per_mwh > 0 and self._plan_export:
            anchor = np.array([self._plan_export.get(t, np.nan) for t in qh_times])
        solution = self.solve_schedule(
            qh_times,
            self.renewable_derate * fc["wind_mw"].to_numpy(float),
            self.renewable_derate * fc["pv_mw"].to_numpy(float),
            price_qh.to_numpy(float),
            soc0_mwh=sim.battery.energy_mwh,
            terminal_mwh=(
                bat.soc_terminal_target * bat.energy_capacity_mwh
                if bat.soc_terminal_target is not None
                else None
            ),
            anchor_export_mw=anchor,
        )
        self._plan_export.update(solution.export_mw.to_dict())
        self._plan_bess.update(solution.bess_mw.to_dict())
        return solution.export_mw.to_dict()

    def act(self, event: MarketEvent, sim: Simulator) -> Action:
        if event.type == EventType.DAA_GATE_CLOSURE:
            target = self._reoptimize(event, sim, "daa")
            orders: dict[DeliveryProduct, float] = {}
            for p in event.products:
                qhs = [qh.start_utc for qh in p.quarter_hours()]
                orders[p] = float(np.mean([target[t] for t in qhs]))
            return AuctionAction(orders)

        if event.type in (
            EventType.IDA1_GATE_CLOSURE,
            EventType.IDA2_GATE_CLOSURE,
            EventType.IDA3_GATE_CLOSURE,
        ):
            target = self._reoptimize(event, sim, event.market)
            orders = {}
            for p in event.products:
                delta = target[p.start_utc] - sim.book.net_position_mw(p)
                if abs(delta) >= 1.0:
                    orders[p] = delta
            return AuctionAction(orders)

        if event.type == EventType.IDC_DECISION:
            if not self.idc_corrections:
                return IdcAction()
            near = tuple(
                p for p in event.products if p.start_utc - event.time_utc <= self.idc_horizon
            )
            if not near:
                return IdcAction()
            times = pd.DatetimeIndex([p.start_utc for p in near])
            fc = self.renewable_forecaster.forecast(event.time_utc, times)
            renewable = self.renewable_derate * (fc["wind_mw"] + fc["pv_mw"])
            orders = {}
            for p in near:
                # keep the planned battery contribution, refresh the renewable part
                target_mw = min(
                    float(renewable[p.start_utc]) + self._plan_bess.get(p.start_utc, 0.0),
                    self.cfg.site.grid.export_limit_mw,
                )
                delta = target_mw - sim.book.net_position_mw(p)
                if abs(delta) >= self.idc_threshold_mw:
                    orders[p] = delta
            return IdcAction(orders)

        if event.type == EventType.PHYSICAL_DISPATCH:
            product = event.products[0]
            row = sim.profiles.loc[product.start_utc]
            renewables = float(row["wind_avail_mw"] + row["pv_avail_mw"])
            position = sim.book.net_position_mw(product)
            return DispatchAction(bess_power_mw=position - renewables)

        raise ValueError(f"unexpected event {event.type}")
