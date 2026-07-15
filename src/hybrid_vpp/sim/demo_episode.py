"""Stage-2 demonstration: one complete episode with manually scripted actions.

Runs the deterministic simulator for one delivery day with hand-picked
market and dispatch actions, then prints the position, cash-flow, and
physical accounting. Edit the CONFIG block and run::

    uv run python -m hybrid_vpp.sim.demo_episode
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from hybrid_vpp.config.models import load_config
from hybrid_vpp.data.site_profiles import load_site_profiles
from hybrid_vpp.data.sqlite_market_data import MarketDataStore
from hybrid_vpp.markets.calendar import EventType
from hybrid_vpp.sim.simulator import (
    AuctionAction,
    DispatchAction,
    IdcAction,
    Simulator,
)

# --------------------------------------------------------------------------
# CONFIG — edit and run as a module (no CLI flags by design)
# --------------------------------------------------------------------------
CONFIG_PATH = Path("configs/default.yaml")
DELIVERY_DAY = "2025-03-15"
DAA_SELL_MW = 50.0  # flat day-ahead sale
IDA1_ADJUST_MW = 10.0  # extra sale in IDA1 for morning hours
IDC_BUYBACK_MW = 5.0  # small IDC buy-back for evening products
BESS_DISCHARGE_EVENING_MW = 20.0  # discharge during evening peak


def run_demo(config_path: Path = CONFIG_PATH, delivery_day: str = DELIVERY_DAY) -> Simulator:
    cfg = load_config(config_path)
    store = MarketDataStore(cfg.data, cfg.markets, cfg.synthetic_market)
    profiles = load_site_profiles(cfg.data, cfg.site, store)
    sim = Simulator(cfg, store, profiles)

    event = sim.start_episode(pd.Timestamp(delivery_day))
    while event is not None:
        if event.type == EventType.DAA_GATE_CLOSURE:
            action = AuctionAction({p: DAA_SELL_MW for p in event.products})
        elif event.type == EventType.IDA1_GATE_CLOSURE:
            morning = [p for p in event.products if 6 <= p.local_start.hour < 12]
            action = AuctionAction({p: IDA1_ADJUST_MW for p in morning})
        elif event.type in (EventType.IDA2_GATE_CLOSURE, EventType.IDA3_GATE_CLOSURE):
            action = AuctionAction()
        elif event.type == EventType.IDC_DECISION:
            orders = {}
            if event.time_utc.hour == 12:  # one scripted buy-back at local ~14:00
                evening = [p for p in event.products if p.local_start.hour in (19, 20)]
                orders = {p: -IDC_BUYBACK_MW for p in evening}
            action = IdcAction(orders)
        elif event.type == EventType.PHYSICAL_DISPATCH:
            hour = event.products[0].local_start.hour
            bess = BESS_DISCHARGE_EVENING_MW if 18 <= hour < 21 else 0.0
            action = DispatchAction(bess_power_mw=bess)
        else:  # pragma: no cover
            raise RuntimeError(f"unexpected decision event {event.type}")
        _, event = sim.step(action)

    report(sim)
    return sim


def report(sim: Simulator) -> None:
    print(f"\n=== episode {sim.delivery_days[0].date()} ===")
    print(f"events processed: {len(sim.events)}")
    print(f"trades: {len(sim.book.trades)}, turnover {sim.book.turnover_mwh():.1f} MWh")

    totals = sim.ledger.by_component()
    print("\ncash flows [EUR]:")
    for component, amount in totals.items():
        if amount:
            print(f"  {component:>20}: {amount:+12.2f}")
    print(f"  {'TOTAL':>20}: {sim.ledger.total():+12.2f}")

    records = sim.dispatch_records.values()
    delivered = sum(r.dispatch.grid_power_mw * r.product.hours for r in records)
    contracted = sum(sim.book.contracted_energy_mwh(r.product) for r in records)
    deviation = sum(s.deviation_mwh for s in sim.settlements.values())
    curtailed = sum(
        (r.dispatch.wind_curtail_mw + r.dispatch.pv_curtail_mw) * r.product.hours for r in records
    )
    corrected = sum(1 for r in records if r.dispatch.was_corrected)
    max_export = max(r.dispatch.grid_power_mw for r in records)
    print("\nphysical accounting:")
    print(f"  delivered:  {delivered:9.2f} MWh")
    print(f"  contracted: {contracted:9.2f} MWh")
    print(f"  deviation:  {deviation:+9.2f} MWh")
    print(f"  curtailed:  {curtailed:9.2f} MWh")
    print(f"  max export: {max_export:9.2f} MW (limit {sim.cfg.site.grid.export_limit_mw})")
    print(f"  intervals with corrected dispatch: {corrected}/{len(sim.dispatch_records)}")
    print(f"  final SoC: {sim.battery.soc:.3f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run_demo()
