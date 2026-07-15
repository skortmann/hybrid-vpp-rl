"""Feasibility projection of requested dispatch onto the physical feasible set.

The hybrid park is intentionally oversized: installed wind + PV capacity may
exceed the grid-connection export limit. This module transforms a requested
dispatch (BESS power, wind curtailment, PV curtailment) into an explicitly
feasible dispatch that satisfies

* box constraints:      p_bess in [p_min, p_max] (SoC- and rating-aware),
                        0 <= curt_w <= wind_avail, 0 <= curt_pv <= pv_avail,
* grid constraint:      -import_limit <= grid_power <= export_limit,

where ``grid_power = wind_avail - curt_w + pv_avail - curt_pv + p_bess``
(positive = export, positive BESS power = discharge).

The final grid power is *derived from the corrected dispatch variables* —
the export value itself is never clipped, so the physical power balance is
preserved exactly. Every correction is recorded with a reason.

Feasibility is structurally guaranteed: with full curtailment and maximum
charge the export constraint is always satisfiable, and with zero
curtailment and maximum discharge the import constraint is always
satisfiable, for any non-negative available generation.

Congestion-resolution modes
---------------------------
``optimization``            weighted least-squares projection (exact KKT
                            solution via bisection on the multiplier),
``battery_first``           use BESS charging headroom before curtailing,
``curtailment_first``       curtail (proportionally) before changing BESS
                            charging; requested discharge is reduced first,
``pv_first`` / ``wind_first``  curtail one technology before the other,
``proportional_curtailment``   split curtailment by available headroom.

All heuristic modes first reduce *discharge* toward zero — discharging into
export congestion is never sensible — and fall back to BESS charging when
curtailment potential is exhausted (only possible with grid import while
charging, i.e. an import-side interaction).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hybrid_vpp.config.models import CongestionResolutionConfig, GridConnectionConfig

_TOL = 1e-9


@dataclass(frozen=True, slots=True)
class ActionCorrection:
    variable: str
    requested: float
    applied: float
    reason: str


@dataclass(slots=True)
class FeasibleDispatch:
    """Feasible dispatch for one interval plus a full correction audit trail."""

    # applied values
    bess_power_mw: float
    wind_curtail_mw: float
    pv_curtail_mw: float
    grid_power_mw: float
    # requested values (raw, before any correction)
    requested_bess_power_mw: float
    requested_wind_curtail_mw: float
    requested_pv_curtail_mw: float
    #: export that would occur with the box-clipped request, before congestion resolution
    unconstrained_grid_power_mw: float
    #: max(0, unconstrained export - export limit)
    excess_export_mw: float
    #: curtailment added by the grid constraint (technical), by technology
    congestion_wind_curtail_mw: float = 0.0
    congestion_pv_curtail_mw: float = 0.0
    #: extra BESS charging power used to absorb excess generation
    congestion_charge_mw: float = 0.0
    corrections: list[ActionCorrection] = field(default_factory=list)

    @property
    def was_corrected(self) -> bool:
        return len(self.corrections) > 0


def project_dispatch(
    *,
    bess_power_mw: float,
    wind_curtail_mw: float,
    pv_curtail_mw: float,
    wind_avail_mw: float,
    pv_avail_mw: float,
    bess_bounds_mw: tuple[float, float],
    grid: GridConnectionConfig,
    resolution: CongestionResolutionConfig | None = None,
) -> FeasibleDispatch:
    """Project a requested dispatch onto the feasible set (see module docstring)."""
    res = resolution or grid.congestion_resolution
    corrections: list[ActionCorrection] = []

    req_b, req_cw, req_cpv = bess_power_mw, wind_curtail_mw, pv_curtail_mw
    p_min, p_max = bess_bounds_mw
    if p_min > _TOL or p_max < -_TOL or p_min > p_max:
        raise ValueError(f"invalid BESS bounds {bess_bounds_mw}: must satisfy p_min <= 0 <= p_max")
    if wind_avail_mw < -_TOL or pv_avail_mw < -_TOL:
        raise ValueError("available generation must be non-negative")
    wind_avail_mw, pv_avail_mw = max(wind_avail_mw, 0.0), max(pv_avail_mw, 0.0)

    # ---- step 1: box constraints (asset feasibility), recorded per variable
    b = _clip(
        req_b,
        p_min,
        p_max,
        "bess_power_mw",
        "BESS power limited by rating / SoC window",
        corrections,
    )
    cw = _clip(
        req_cw,
        0.0,
        wind_avail_mw,
        "wind_curtail_mw",
        "wind curtailment limited to [0, available]",
        corrections,
    )
    cpv = _clip(
        req_cpv,
        0.0,
        pv_avail_mw,
        "pv_curtail_mw",
        "PV curtailment limited to [0, available]",
        corrections,
    )

    def grid_power(b_: float, cw_: float, cpv_: float) -> float:
        return wind_avail_mw - cw_ + pv_avail_mw - cpv_ + b_

    unconstrained = grid_power(b, cw, cpv)
    export_limit = grid.export_limit_mw
    import_limit = grid.import_limit_mw if grid.allow_grid_import else 0.0
    excess = max(0.0, unconstrained - export_limit)
    deficit = max(0.0, -import_limit - unconstrained)

    if excess > _TOL or deficit > _TOL:
        if res.mode == "optimization":
            b, cw, cpv = _project_qp(
                (b, cw, cpv),
                (p_min, p_max),
                wind_avail_mw,
                pv_avail_mw,
                export_limit,
                import_limit,
                res,
                base=wind_avail_mw + pv_avail_mw,
            )
        else:
            b, cw, cpv = _resolve_heuristic(
                res.mode,
                b,
                cw,
                cpv,
                p_min,
                p_max,
                wind_avail_mw,
                pv_avail_mw,
                export_limit,
                import_limit,
            )
        side = "export above grid limit" if excess > _TOL else "import above grid limit"
        for name, before, after in (
            ("bess_power_mw", req_b, b),
            ("wind_curtail_mw", req_cw, cw),
            ("pv_curtail_mw", req_cpv, cpv),
        ):
            if abs(after - before) > _TOL and not any(
                c.variable == name and abs(c.applied - after) <= _TOL for c in corrections
            ):
                corrections.append(
                    ActionCorrection(
                        name, before, after, f"grid congestion resolution ({res.mode}): {side}"
                    )
                )

    final = grid_power(b, cw, cpv)
    if final > export_limit + 1e-6 or final < -import_limit - 1e-6:
        raise AssertionError(
            f"feasibility projection failed: grid={final:.6f} MW, "
            f"limits=[-{import_limit}, {export_limit}]"
        )

    # congestion-attributed corrections (technical vs. economic curtailment)
    boxed_cw = min(max(req_cw, 0.0), wind_avail_mw)
    boxed_cpv = min(max(req_cpv, 0.0), pv_avail_mw)
    boxed_b = min(max(req_b, p_min), p_max)
    return FeasibleDispatch(
        bess_power_mw=b,
        wind_curtail_mw=cw,
        pv_curtail_mw=cpv,
        grid_power_mw=final,
        requested_bess_power_mw=req_b,
        requested_wind_curtail_mw=req_cw,
        requested_pv_curtail_mw=req_cpv,
        unconstrained_grid_power_mw=unconstrained,
        excess_export_mw=excess,
        congestion_wind_curtail_mw=max(0.0, cw - boxed_cw),
        congestion_pv_curtail_mw=max(0.0, cpv - boxed_cpv),
        congestion_charge_mw=max(0.0, boxed_b - b),
        corrections=corrections,
    )


def _clip(
    value: float,
    lo: float,
    hi: float,
    name: str,
    reason: str,
    corrections: list[ActionCorrection],
) -> float:
    clipped = min(max(value, lo), hi)
    if abs(clipped - value) > _TOL:
        corrections.append(ActionCorrection(name, value, clipped, reason))
    return clipped


# ------------------------------------------------------------- optimization mode


def _project_qp(
    x0: tuple[float, float, float],
    bess_bounds: tuple[float, float],
    wind_avail: float,
    pv_avail: float,
    export_limit: float,
    import_limit: float,
    res: CongestionResolutionConfig,
    base: float,
) -> tuple[float, float, float]:
    """Exact weighted projection onto the grid constraint.

    minimize  w_b (b-b0)^2 + w_w (cw-cw0)^2 + w_pv (cpv-cpv0)^2
    s.t.      box constraints,  -import_limit <= base + b - cw - cpv <= export_limit

    KKT stationarity gives x_i(lam) = clip_box(x0_i - lam * a_i / w_i) with
    a = (+1, -1, -1); the grid power is continuous and monotonically
    non-increasing in lam, so the active multiplier is found by bisection.
    """
    w = (res.weights.bess_deviation, res.weights.wind_curtailment, res.weights.pv_curtailment)
    lo = (bess_bounds[0], 0.0, 0.0)
    hi = (bess_bounds[1], wind_avail, pv_avail)
    a = (1.0, -1.0, -1.0)

    def x_of(lam: float) -> tuple[float, float, float]:
        return tuple(min(max(x0[i] - lam * a[i] / w[i], lo[i]), hi[i]) for i in range(3))  # type: ignore[return-value]

    def g_of(lam: float) -> float:
        x = x_of(lam)
        return base + x[0] - x[1] - x[2]

    g0 = g_of(0.0)
    if g0 > export_limit:
        target, sign = export_limit, +1.0
    elif g0 < -import_limit:
        target, sign = -import_limit, -1.0
    else:
        return x_of(0.0)

    # bracket the multiplier, then bisect (g is monotone non-increasing in lam)
    lam_hi = sign
    for _ in range(80):
        if sign * (g_of(lam_hi) - target) <= 0:
            break
        lam_hi *= 2.0
    lam_lo = 0.0
    for _ in range(200):
        mid = 0.5 * (lam_lo + lam_hi)
        if sign * (g_of(mid) - target) > 0:
            lam_lo = mid
        else:
            lam_hi = mid
        if abs(lam_hi - lam_lo) < 1e-13 * max(1.0, abs(lam_hi)):
            break
    return x_of(lam_hi)


# --------------------------------------------------------------- heuristic modes


def _resolve_heuristic(
    mode: str,
    b: float,
    cw: float,
    cpv: float,
    p_min: float,
    p_max: float,
    wind_avail: float,
    pv_avail: float,
    export_limit: float,
    import_limit: float,
) -> tuple[float, float, float]:
    def grid(b_: float, cw_: float, cpv_: float) -> float:
        return wind_avail - cw_ + pv_avail - cpv_ + b_

    # --- import-side violation: reduce charging, then reduce curtailment
    deficit = -import_limit - grid(b, cw, cpv)
    if deficit > _TOL:
        step = min(deficit, max(0.0, -b))  # reduce charge magnitude toward 0
        b += step
        deficit -= step
        for name in ("cw", "cpv"):
            if deficit <= _TOL:
                break
            if name == "cw":
                step = min(deficit, cw)
                cw -= step
            else:
                step = min(deficit, cpv)
                cpv -= step
            deficit -= step
        return b, cw, cpv

    # --- export-side congestion
    excess = grid(b, cw, cpv) - export_limit
    if excess <= _TOL:
        return b, cw, cpv

    # all modes: discharging into congestion is reduced first
    step = min(excess, max(b, 0.0))
    b -= step
    excess -= step

    def charge(excess_: float, b_: float) -> tuple[float, float]:
        step_ = min(excess_, b_ - p_min)
        return excess_ - step_, b_ - step_

    def curtail_prop(excess_: float, cw_: float, cpv_: float) -> tuple[float, float, float]:
        head_w, head_pv = wind_avail - cw_, pv_avail - cpv_
        head = head_w + head_pv
        if head <= _TOL:
            return excess_, cw_, cpv_
        take = min(excess_, head)
        cw_ += take * head_w / head
        cpv_ += take * head_pv / head
        return excess_ - take, cw_, cpv_

    def curtail_one(excess_: float, cur: float, avail: float) -> tuple[float, float]:
        step_ = min(excess_, avail - cur)
        return excess_ - step_, cur + step_

    if mode == "battery_first":
        excess, b = charge(excess, b)
        excess, cw, cpv = curtail_prop(excess, cw, cpv)
    elif mode in ("curtailment_first", "proportional_curtailment"):
        excess, cw, cpv = curtail_prop(excess, cw, cpv)
        excess, b = charge(excess, b)
    elif mode == "pv_first":
        excess, cpv = curtail_one(excess, cpv, pv_avail)
        excess, cw = curtail_one(excess, cw, wind_avail)
        excess, b = charge(excess, b)
    elif mode == "wind_first":
        excess, cw = curtail_one(excess, cw, wind_avail)
        excess, cpv = curtail_one(excess, cpv, pv_avail)
        excess, b = charge(excess, b)
    else:  # pragma: no cover
        raise ValueError(f"unknown congestion resolution mode {mode!r}")
    return b, cw, cpv
