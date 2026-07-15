"""Feasibility projection: oversized park behind a constrained grid connection.

Covers required cases 1-5 and 7 (case 6 = settlement, case 8 = DST live in
integration tests). Site: 70 MW wind + 50 MW PV behind X = 100 MW export.
"""

import pytest

from hybrid_vpp.assets.feasibility import project_dispatch
from hybrid_vpp.config.models import (
    CongestionResolutionConfig,
    CongestionWeights,
    GridConnectionConfig,
)

X = 100.0  # export limit


def grid_cfg(mode="optimization", import_mw=30.0, **weights) -> GridConnectionConfig:
    return GridConnectionConfig(
        export_limit_mw=X,
        import_limit_mw=import_mw,
        allow_grid_import=import_mw > 0,
        congestion_resolution=CongestionResolutionConfig(
            mode=mode, weights=CongestionWeights(**weights)
        ),
    )


def project(
    mode="optimization", b=0.0, cw=0.0, cpv=0.0, w=70.0, pv=50.0, bounds=(-30.0, 30.0), grid=None
):
    return project_dispatch(
        bess_power_mw=b,
        wind_curtail_mw=cw,
        pv_curtail_mw=cpv,
        wind_avail_mw=w,
        pv_avail_mw=pv,
        bess_bounds_mw=bounds,
        grid=grid or grid_cfg(mode),
    )


def balance_ok(r, w=70.0, pv=50.0):
    assert r.grid_power_mw == pytest.approx(
        w - r.wind_curtail_mw + pv - r.pv_curtail_mw + r.bess_power_mw, abs=1e-9
    )


# --- case 1: generation below the limit -> request passes through unchanged
@pytest.mark.parametrize(
    "mode",
    [
        "optimization",
        "battery_first",
        "curtailment_first",
        "pv_first",
        "wind_first",
        "proportional_curtailment",
    ],
)
def test_case1_no_congestion_passthrough(mode):
    r = project(mode, b=10.0, w=50.0, pv=30.0)
    assert not r.was_corrected
    assert (r.bess_power_mw, r.wind_curtail_mw, r.pv_curtail_mw) == (10.0, 0.0, 0.0)
    assert r.grid_power_mw == pytest.approx(90.0)
    balance_ok(r, w=50.0, pv=30.0)


# --- case 2: excess absorbed by BESS charging
def test_case2_bess_absorbs_excess():
    r = project("battery_first")  # 120 MW available, X=100 -> 20 MW excess
    assert r.bess_power_mw == pytest.approx(-20.0)
    assert r.wind_curtail_mw == pytest.approx(0.0)
    assert r.pv_curtail_mw == pytest.approx(0.0)
    assert r.grid_power_mw == pytest.approx(X)
    assert r.excess_export_mw == pytest.approx(20.0)
    assert r.congestion_charge_mw == pytest.approx(20.0)
    assert r.was_corrected
    balance_ok(r)


# --- case 3: full BESS -> remaining excess curtailed, never beyond available
def test_case3_full_bess_curtails():
    r = project("battery_first", bounds=(0.0, 30.0))  # no charging headroom
    assert r.bess_power_mw == pytest.approx(0.0)
    total_curt = r.wind_curtail_mw + r.pv_curtail_mw
    assert total_curt == pytest.approx(20.0)
    assert 0 <= r.wind_curtail_mw <= 70.0 and 0 <= r.pv_curtail_mw <= 50.0
    assert r.grid_power_mw == pytest.approx(X)
    assert r.congestion_wind_curtail_mw + r.congestion_pv_curtail_mw == pytest.approx(20.0)
    balance_ok(r)


# --- case 4: discharge requested during congestion is rejected/reduced
@pytest.mark.parametrize("mode", ["optimization", "battery_first", "curtailment_first"])
def test_case4_discharge_reduced_under_congestion(mode):
    r = project(mode, b=25.0)  # 120 avail + 25 discharge -> 45 MW over the limit
    assert r.bess_power_mw < 25.0
    assert r.grid_power_mw <= X + 1e-6
    assert any(c.variable == "bess_power_mw" for c in r.corrections)
    balance_ok(r)


# --- case 5: very large excess -> charging and curtailment combine, balance holds
def test_case5_large_excess_combined():
    grid = grid_cfg("battery_first")
    r = project_dispatch(
        bess_power_mw=0.0,
        wind_curtail_mw=0.0,
        pv_curtail_mw=0.0,
        wind_avail_mw=70.0,
        pv_avail_mw=50.0,
        bess_bounds_mw=(-10.0, 30.0),
        grid=grid,
    )
    assert r.bess_power_mw == pytest.approx(-10.0)  # all charging headroom used
    assert r.wind_curtail_mw + r.pv_curtail_mw == pytest.approx(10.0)
    assert r.grid_power_mw == pytest.approx(X, abs=1e-6)
    balance_ok(r)


# --- case 7: curtailment priority modes
def test_case7_pv_first():
    r = project("pv_first", bounds=(0.0, 30.0))
    assert r.pv_curtail_mw == pytest.approx(20.0)
    assert r.wind_curtail_mw == pytest.approx(0.0)


def test_case7_wind_first():
    r = project("wind_first", bounds=(0.0, 30.0))
    assert r.wind_curtail_mw == pytest.approx(20.0)
    assert r.pv_curtail_mw == pytest.approx(0.0)


def test_case7_proportional():
    r = project("proportional_curtailment", bounds=(0.0, 30.0))
    # headroom 70 wind / 50 pv -> 20 MW split 70:50
    assert r.wind_curtail_mw == pytest.approx(20.0 * 70.0 / 120.0)
    assert r.pv_curtail_mw == pytest.approx(20.0 * 50.0 / 120.0)


def test_pv_first_overflows_to_wind():
    grid = grid_cfg("pv_first")
    r = project_dispatch(
        bess_power_mw=0.0,
        wind_curtail_mw=0.0,
        pv_curtail_mw=0.0,
        wind_avail_mw=120.0,
        pv_avail_mw=50.0,
        bess_bounds_mw=(0.0, 30.0),
        grid=grid,
    )  # 170 avail, 70 excess: PV gives 50, wind the remaining 20
    assert r.pv_curtail_mw == pytest.approx(50.0)
    assert r.wind_curtail_mw == pytest.approx(20.0)
    assert r.grid_power_mw == pytest.approx(X, abs=1e-6)


# --- optimization mode: weights steer the split, solution is exact
def test_optimization_prefers_cheap_bess():
    # exact KKT with w=(1,10,10): x_i = -lam*a_i/w_i, constraint -1.2*lam = -20
    # -> lam = 50/3: b = -50/3, cw = cpv = 5/3 (quadratic costs share the load)
    r = project("optimization")
    assert r.bess_power_mw == pytest.approx(-50.0 / 3.0, abs=1e-6)
    assert r.wind_curtail_mw == pytest.approx(5.0 / 3.0, abs=1e-6)
    assert r.pv_curtail_mw == pytest.approx(5.0 / 3.0, abs=1e-6)
    assert r.grid_power_mw == pytest.approx(X, abs=1e-6)


def test_optimization_weights_flip_priority():
    grid = grid_cfg("optimization", bess_deviation=100.0, wind_curtailment=1.0, pv_curtailment=1.0)
    r = project(grid=grid)
    assert abs(r.bess_power_mw) < 1.0  # battery barely moves
    assert r.wind_curtail_mw + r.pv_curtail_mw == pytest.approx(
        20.0 - abs(r.bess_power_mw), abs=1e-6
    )
    assert r.grid_power_mw == pytest.approx(X, abs=1e-6)


def test_optimization_exact_kkt_split():
    # equal weights, no battery: curtailment splits equally between wind and PV
    grid = grid_cfg("optimization", bess_deviation=1e9, wind_curtailment=1.0, pv_curtailment=1.0)
    r = project(grid=grid)
    assert r.wind_curtail_mw == pytest.approx(10.0, abs=1e-4)
    assert r.pv_curtail_mw == pytest.approx(10.0, abs=1e-4)


# --- import-side constraint
def test_import_limit_enforced():
    # request 30 MW charge with zero generation and import capped at 10 MW
    grid = GridConnectionConfig(export_limit_mw=X, import_limit_mw=10.0, allow_grid_import=True)
    r = project_dispatch(
        bess_power_mw=-30.0,
        wind_curtail_mw=0.0,
        pv_curtail_mw=0.0,
        wind_avail_mw=0.0,
        pv_avail_mw=0.0,
        bess_bounds_mw=(-30.0, 30.0),
        grid=grid,
    )
    assert r.grid_power_mw == pytest.approx(-10.0, abs=1e-6)
    assert r.bess_power_mw == pytest.approx(-10.0, abs=1e-6)


def test_no_import_allowed():
    grid = GridConnectionConfig(export_limit_mw=X, import_limit_mw=0.0, allow_grid_import=False)
    r = project_dispatch(
        bess_power_mw=-30.0,
        wind_curtail_mw=0.0,
        pv_curtail_mw=0.0,
        wind_avail_mw=5.0,
        pv_avail_mw=0.0,
        bess_bounds_mw=(-30.0, 30.0),
        grid=grid,
    )
    assert r.grid_power_mw >= -1e-9
    assert r.bess_power_mw == pytest.approx(-5.0, abs=1e-6)  # only renewable charging


# --- audit trail
def test_corrections_recorded_with_reasons():
    # cw=-5 violates its box; b=25 on top of 120 MW available violates the grid
    r = project("optimization", b=25.0, cw=-5.0, cpv=0.0)
    variables = {c.variable for c in r.corrections}
    assert {"wind_curtail_mw", "pv_curtail_mw", "bess_power_mw"} <= variables
    assert all(c.reason for c in r.corrections)
    assert r.requested_bess_power_mw == 25.0
    assert r.requested_wind_curtail_mw == -5.0
    assert r.requested_pv_curtail_mw == 0.0
    assert r.grid_power_mw <= X + 1e-6
