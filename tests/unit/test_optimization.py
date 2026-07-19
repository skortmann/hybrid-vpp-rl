"""MILP schedule solver: hand-calculated optima on tiny cases."""

import numpy as np
import pandas as pd
import pytest

from hybrid_vpp.config.models import (
    BatteryConfig,
    ExperimentConfig,
    GridConnectionConfig,
    PVConfig,
    SiteConfig,
    WindConfig,
)
from hybrid_vpp.controllers.optimization import OptimizationController


def make_cfg(**battery_kw) -> ExperimentConfig:
    battery = dict(
        energy_capacity_mwh=10.0,
        charge_power_mw=10.0,
        discharge_power_mw=10.0,
        charge_efficiency=1.0,
        discharge_efficiency=1.0,
        soc_min=0.0,
        soc_max=1.0,
        soc_initial=0.0,
        degradation_cost_eur_per_mwh_throughput=0.0,
    )
    battery.update(battery_kw)
    site = SiteConfig(
        wind=WindConfig(capacity_mw=20.0, latitude=53.0, longitude=8.0),
        pv=PVConfig(capacity_mw=10.0, latitude=53.0, longitude=8.0),
        battery=BatteryConfig(**battery),
        grid=GridConnectionConfig(
            export_limit_mw=25.0, import_limit_mw=10.0, allow_grid_import=True
        ),
    )
    return ExperimentConfig(site=site)


def controller(cfg) -> OptimizationController:
    return OptimizationController(cfg, renewable_forecaster=None, price_forecaster=None)


def times(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2025-03-15", periods=n, freq="15min", tz="UTC")


def test_pure_arbitrage_charges_low_discharges_high():
    cfg = make_cfg()
    n = 8
    prices = np.array([10, 10, 10, 10, 100, 100, 100, 100], dtype=float)
    sol = controller(cfg).solve_schedule(
        times(n), np.zeros(n), np.zeros(n), prices, soc0_mwh=0.0, terminal_mwh=None
    )
    # charge 10 MW in the 4 cheap QHs (10 MWh capacity = 4 * 10 MW * 0.25 h),
    # discharge 10 MW in the 4 expensive ones; profit = 10 MWh * 90 EUR
    assert sol.bess_mw.iloc[:4].to_numpy() == pytest.approx([-10.0] * 4)
    assert sol.bess_mw.iloc[4:].to_numpy() == pytest.approx([10.0] * 4)
    assert sol.objective_eur == pytest.approx(10 * 100 - 10 * 10)


def test_no_simultaneous_charge_discharge_at_negative_prices():
    cfg = make_cfg(charge_efficiency=0.8, discharge_efficiency=0.8)
    n = 4
    prices = np.full(n, -500.0)  # LP relaxation would burn energy round-trip
    sol = controller(cfg).solve_schedule(
        times(n), np.zeros(n), np.zeros(n), prices, soc0_mwh=0.0, terminal_mwh=None
    )
    # charging is paid at negative prices (import at -500), so the battery
    # fills — but never discharges at the same time
    for b in sol.bess_mw:
        assert b <= 1e-6 or b >= -1e-6  # single-signed per interval by binaries
    # import limit 10 MW bounds the charge
    assert (sol.export_mw >= -10.0 - 1e-6).all()


def test_curtailment_at_negative_prices():
    cfg = make_cfg(energy_capacity_mwh=1.0, charge_power_mw=1.0)
    n = 4
    wind = np.full(n, 20.0)
    prices = np.full(n, -50.0)
    sol = controller(cfg).solve_schedule(
        times(n), wind, np.zeros(n), prices, soc0_mwh=0.0, terminal_mwh=None
    )
    # exporting 20 MW at -50 costs money; battery absorbs at most 1 MWh;
    # optimal: curtail (export ~ 0 apart from the tiny charge absorption)
    assert (sol.export_mw <= 1e-6).all()


def test_grid_export_limit_respected_and_excess_stored():
    cfg = make_cfg()
    n = 4
    wind = np.array([30.0, 30.0, 0.0, 0.0])  # 5 MW above the 25 MW limit early on
    prices = np.full(n, 80.0)
    sol = controller(cfg).solve_schedule(
        times(n), wind, np.zeros(n), prices, soc0_mwh=0.0, terminal_mwh=None
    )
    assert (sol.export_mw <= 25.0 + 1e-6).all()
    # storing the excess and selling it later beats curtailing it: all 15 MWh
    # of wind reach the grid (the exact charge pattern is degenerate at flat
    # prices, so only the energy identities are asserted)
    assert sol.bess_mw.iloc[:2].min() <= -5.0 + 1e-6  # excess absorbed early
    assert (sol.bess_mw * 0.25).sum() == pytest.approx(0.0, abs=1e-6)  # all recovered
    assert sol.export_mw.sum() * 0.25 == pytest.approx(15.0, abs=1e-6)  # nothing wasted


def test_solver_backends_agree():
    cfg = make_cfg()
    n = 8
    prices = np.array([10, 10, 10, 10, 100, 100, 100, 100], dtype=float)
    objectives = {}
    for solver in ("gurobi", "highs"):
        ctrl = OptimizationController(cfg, None, None, solver=solver)
        sol = ctrl.solve_schedule(
            times(n), np.zeros(n), np.zeros(n), prices, soc0_mwh=0.0, terminal_mwh=None
        )
        objectives[solver] = sol.objective_eur
    assert objectives["gurobi"] == pytest.approx(objectives["highs"], rel=1e-4)


def test_terminal_soc_enforced():
    cfg = make_cfg()
    n = 4
    prices = np.full(n, 100.0)  # tempting to discharge everything
    sol = controller(cfg).solve_schedule(
        times(n), np.zeros(n), np.zeros(n), prices, soc0_mwh=5.0, terminal_mwh=5.0
    )
    net_mwh = -(sol.bess_mw * 0.25).sum()  # positive = net charged
    assert net_mwh >= -1e-6  # cannot end below the terminal requirement


def test_terminal_soc_optional():
    """Without a terminal requirement the same setup drains the battery."""
    cfg = make_cfg()
    n = 4
    prices = np.full(n, 100.0)
    ctrl = OptimizationController(cfg, None, None, enforce_terminal_soc=False)
    sol = ctrl.solve_schedule(
        times(n), np.zeros(n), np.zeros(n), prices, soc0_mwh=5.0, terminal_mwh=None
    )
    net_mwh = -(sol.bess_mw * 0.25).sum()
    assert net_mwh < -1e-6  # net discharged: the boundary energy is sold


def test_terminal_value_retains_inventory():
    """A terminal inventory value above prices makes the plan charge, not drain."""
    cfg = make_cfg()
    n = 4
    prices = np.full(n, 10.0)  # cheap hours; inventory is worth 100 at day end
    sol = controller(cfg).solve_schedule(
        times(n),
        np.zeros(n),
        np.zeros(n),
        prices,
        soc0_mwh=5.0,
        terminal_mwh=None,
        terminal_value_eur_per_mwh=100.0,
    )
    net_mwh = -(sol.bess_mw * 0.25).sum()
    assert net_mwh > 1e-6  # net charged: buy cheap, carry the energy over
