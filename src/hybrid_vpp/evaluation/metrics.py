"""Episode and multi-episode evaluation metrics.

Covers economics, physical behavior (incl. the oversized-park /
grid-congestion metrics), and trading behavior. All energies in MWh,
powers in MW, cash in EUR.
"""

from __future__ import annotations

import pandas as pd

from hybrid_vpp.core.timegrid import energy_mwh
from hybrid_vpp.sim.simulator import Simulator


def episode_metrics(sim: Simulator) -> dict[str, float]:
    """KPIs of the episode currently held by the simulator."""
    site = sim.cfg.site
    records = list(sim.dispatch_records.values())
    settlements = list(sim.settlements.values())
    ledger = sim.ledger.by_component()
    hours = records[0].product.hours if records else 0.25

    wind_avail = sum(r.wind_avail_mw for r in records) * hours
    pv_avail = sum(r.pv_avail_mw for r in records) * hours
    wind_curt = sum(r.dispatch.wind_curtail_mw for r in records) * hours
    pv_curt = sum(r.dispatch.pv_curtail_mw for r in records) * hours
    congestion_wind = sum(r.dispatch.congestion_wind_curtail_mw for r in records) * hours
    congestion_pv = sum(r.dispatch.congestion_pv_curtail_mw for r in records) * hours
    congestion_charge = sum(r.dispatch.congestion_charge_mw for r in records) * hours
    charged = sum(r.battery.charge_energy_mwh for r in records)
    discharged = sum(r.battery.discharge_energy_mwh for r in records)
    export = sum(max(r.dispatch.grid_power_mw, 0.0) for r in records) * hours
    imports = sum(-min(r.dispatch.grid_power_mw, 0.0) for r in records) * hours
    delivered = sum(energy_mwh(r.dispatch.grid_power_mw, r.product.duration) for r in records)
    contracted = sum(s.contracted_mwh for s in settlements)
    deviation = sum(abs(s.deviation_mwh) for s in settlements)
    limit = site.grid.export_limit_mw

    exceed_intervals = sum(1 for r in records if r.dispatch.unconstrained_grid_power_mw > limit)
    excess_energy = sum(r.dispatch.excess_export_mw for r in records) * hours
    near_limit = sum(1 for r in records if r.dispatch.grid_power_mw >= 0.98 * limit)
    renewable_gen = wind_avail + pv_avail - wind_curt - pv_curt

    market_revenue = sum(ledger[m] for m in ("daa", "ida1", "ida2", "ida3", "idc"))
    total = sim.ledger.total()
    return {
        # economics
        "total_net_revenue_eur": total,
        "daa_revenue_eur": ledger["daa"],
        "ida1_revenue_eur": ledger["ida1"],
        "ida2_revenue_eur": ledger["ida2"],
        "ida3_revenue_eur": ledger["ida3"],
        "idc_revenue_eur": ledger["idc"],
        "market_revenue_eur": market_revenue,
        "imbalance_cash_eur": ledger["imbalance"],
        "transaction_cost_eur": ledger["transaction_cost"],
        "degradation_cost_eur": ledger["degradation"],
        "curtailment_penalty_eur": ledger["curtailment_penalty"],
        "revenue_per_mw_installed_eur": total / site.installed_generation_mw,
        "revenue_per_mwh_renewable_eur": total / renewable_gen if renewable_gen else 0.0,
        # physical
        "wind_available_mwh": wind_avail,
        "pv_available_mwh": pv_avail,
        "wind_curtailed_mwh": wind_curt,
        "pv_curtailed_mwh": pv_curt,
        "curtailment_ratio": (wind_curt + pv_curt) / (wind_avail + pv_avail)
        if wind_avail + pv_avail
        else 0.0,
        "bess_charged_mwh": charged,
        "bess_discharged_mwh": discharged,
        "equivalent_full_cycles": (charged + discharged) / (2 * site.battery.energy_capacity_mwh),
        "grid_export_mwh": export,
        "grid_import_mwh": imports,
        "delivered_mwh": delivered,
        "contracted_mwh": contracted,
        "abs_deviation_mwh": deviation,
        # oversizing / congestion
        "oversizing_ratio": site.oversizing_ratio,
        "excess_capacity_mw": site.excess_capacity_mw,
        "intervals_unconstrained_above_limit": float(exceed_intervals),
        "excess_energy_before_correction_mwh": excess_energy,
        "congestion_charge_mwh": congestion_charge,
        "congestion_wind_curtailed_mwh": congestion_wind,
        "congestion_pv_curtailed_mwh": congestion_pv,
        "economic_curtailment_mwh": (wind_curt + pv_curt) - (congestion_wind + congestion_pv),
        "grid_utilization": export / (limit * len(records) * hours) if records else 0.0,
        "intervals_near_limit": float(near_limit),
        # trading
        "daa_volume_mwh": sim.book.turnover_mwh("daa"),
        "ida1_volume_mwh": sim.book.turnover_mwh("ida1"),
        "ida2_volume_mwh": sim.book.turnover_mwh("ida2"),
        "ida3_volume_mwh": sim.book.turnover_mwh("ida3"),
        "idc_volume_mwh": sim.book.turnover_mwh("idc"),
        "trade_count": float(len(sim.book.trades)),
        "corrected_dispatch_intervals": float(sum(1 for r in records if r.dispatch.was_corrected)),
        "final_soc": sim.battery.soc,
    }


def metrics_frame(per_day: dict[str, dict[str, float]]) -> pd.DataFrame:
    """day -> metrics dict, as a DataFrame with a summary row."""
    df = pd.DataFrame(per_day).T
    df.loc["TOTAL"] = df.sum()
    for col in (
        "final_soc",
        "oversizing_ratio",
        "excess_capacity_mw",
        "curtailment_ratio",
        "grid_utilization",
        "revenue_per_mw_installed_eur",
        "revenue_per_mwh_renewable_eur",
    ):
        if col in df.columns:
            df.loc["TOTAL", col] = df[col].iloc[:-1].mean()
    return df
