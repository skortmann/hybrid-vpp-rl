"""Database inspection: coverage and sanity report for all required series.

Prints table coverage, per-series date ranges, DST-day handling, and gap
counts for the study window. Read-only. Edit CONFIG and run::

    uv run python -m hybrid_vpp.data.inspect_database
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pandas as pd

from hybrid_vpp.config.models import load_config
from hybrid_vpp.core.timegrid import MARKET_TZ
from hybrid_vpp.data.sqlite_market_data import MarketDataStore


def inspect(config_path: Path) -> pd.DataFrame:
    cfg = load_config(config_path)
    store = MarketDataStore(cfg.data, cfg.markets)

    with sqlite3.connect(f"file:{cfg.data.market_db_path}?mode=ro", uri=True) as con:
        n_tables = con.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'").fetchone()[
            0
        ]
    print(f"database: {cfg.data.market_db_path} ({n_tables} tables, read-only)\n")

    series: dict[str, pd.Series | pd.DataFrame] = {
        "daa_prices": store.daa_prices()["price_eur_per_mwh"],
        "ida1_prices": store.ida_prices("ida1"),
        "ida2_prices": store.ida_prices("ida2"),
        "ida3_prices": store.ida_prices("ida3"),
        "idc_ID1": store.idc_indices()["ID1"].dropna(),
        "idc_IDFULL": store.idc_indices()["IDFULL"].dropna(),
        "rebap": store.rebap(),
        "zone_wind_onshore_actual": store.zone_renewables()["actual_wind_onshore"].dropna(),
        "zone_solar_fc_da": store.zone_renewables()["fc_da_solar"].dropna(),
    }

    window = (cfg.split.train_start.tz_localize("UTC"), cfg.split.test_end.tz_localize("UTC"))
    rows = {}
    for name, s in series.items():
        idx = s.index
        in_window = idx[(idx >= window[0]) & (idx < window[1])]
        expected_days = (window[1] - window[0]).days
        rows[name] = {
            "first": idx.min(),
            "last": idx.max(),
            "points": len(idx),
            "days_in_study_window": in_window.normalize().nunique(),
            "expected_days": expected_days,
        }
    report = pd.DataFrame(rows).T
    print(report.to_string())

    print("\nDST handling (spring 2025-03-30 / autumn 2024-10-27, IDA1 products):")
    ida1 = series["ida1_prices"]
    for day, expected in (("2025-03-30", 92), ("2024-10-27", 100)):
        b0 = pd.Timestamp(day).tz_localize(MARKET_TZ).tz_convert("UTC")
        b1 = (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize(MARKET_TZ).tz_convert("UTC")
        n = len(ida1[(ida1.index >= b0) & (ida1.index < b1)])
        status = "OK" if n == expected else "MISMATCH"
        print(f"  {day}: {n} quarter-hours (expected {expected}) {status}")
    return report


# --------------------------------------------------------------------------
# CONFIG — edit and run as a module (no CLI flags by design)
# --------------------------------------------------------------------------
CONFIG_PATH = Path("configs/default.yaml")

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    inspect(CONFIG_PATH)
