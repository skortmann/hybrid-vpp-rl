"""Diagnostic plots (matplotlib, saved as PNG).

Conventions: one entity = one fixed color everywhere (validated
colorblind-safe palette), one y-axis per panel (never dual axes), physical
units in every label, recessive grids, constraint lines in neutral gray.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hybrid_vpp.core.timegrid import energy_mwh
from hybrid_vpp.sim.simulator import Simulator

# fixed entity -> color map (validated categorical palette, light mode)
C = {
    "wind": "#2a78d6",  # blue
    "pv": "#eda100",  # yellow
    "bess": "#e87ba4",  # magenta
    "grid": "#008300",  # green
    "price": "#1baf7a",  # aqua
    "position": "#eb6834",  # orange
    "imbalance": "#4a3aa7",  # violet
    "curtail": "#e34948",  # red (also status: violation)
    "limit": "#52514e",  # neutral ink for constraint lines
}

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
        "legend.frameon": False,
    }
)


def _episode_frame(sim: Simulator) -> pd.DataFrame:
    rows = {}
    for start, r in sorted(sim.dispatch_records.items()):
        s = sim.settlements.get(start)
        rows[start] = {
            "wind_avail": r.wind_avail_mw,
            "pv_avail": r.pv_avail_mw,
            "wind_curt": r.dispatch.wind_curtail_mw,
            "pv_curt": r.dispatch.pv_curtail_mw,
            "bess": r.dispatch.bess_power_mw,
            "grid": r.dispatch.grid_power_mw,
            "grid_unconstrained": r.dispatch.unconstrained_grid_power_mw,
            "soc": r.battery.energy_after_mwh / sim.battery.capacity_mwh,
            "position": sim.book.net_position_mw_at(start),
            "delivered_mwh": energy_mwh(r.dispatch.grid_power_mw, r.product.duration),
            "deviation_mwh": s.deviation_mwh if s else np.nan,
            "rebap": s.settlement_price_eur_per_mwh if s else np.nan,
            "congestion_charge": r.dispatch.congestion_charge_mw,
            "congestion_curt": r.dispatch.congestion_wind_curtail_mw
            + r.dispatch.congestion_pv_curtail_mw,
            "corrected": float(r.dispatch.was_corrected),
        }
    df = pd.DataFrame(rows).T
    df.index = pd.DatetimeIndex(df.index).tz_convert("Europe/Berlin")
    return df


def plot_dispatch_day(sim: Simulator, out: Path) -> Path:
    """Generation, curtailment, grid power vs. position, BESS, prices."""
    df = _episode_frame(sim)
    limit = sim.cfg.site.grid.export_limit_mw
    day = df.index[0].date()
    fig, axes = plt.subplots(
        5,
        1,
        figsize=(10, 12.5),
        sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.0, 0.8, 0.5, 1.0]},
    )

    ax = axes[0]
    ax.stackplot(
        df.index,
        df.wind_avail - df.wind_curt,
        df.pv_avail - df.pv_curt,
        colors=[C["wind"], C["pv"]],
        labels=["wind (used)", "PV (used)"],
        alpha=0.75,
    )
    curt = df.wind_curt + df.pv_curt
    ax.fill_between(
        df.index,
        df.wind_avail + df.pv_avail - curt,
        df.wind_avail + df.pv_avail,
        color=C["curtail"],
        alpha=0.55,
        label="curtailed",
    )
    ax.axhline(limit, color=C["limit"], ls="--", lw=1.2, label=f"grid limit {limit:.0f} MW")
    ax.set_ylim(top=max(limit, (df.wind_avail + df.pv_avail).max()) * 1.25)
    ax.set_ylabel("available power [MW]")
    ax.legend(loc="upper left", ncols=4)
    ax.set_title(f"Hybrid VPP dispatch — {day}")

    ax = axes[1]
    ax.plot(df.index, df.grid, color=C["grid"], lw=1.8, label="grid export")
    ax.plot(df.index, df.position, color=C["position"], lw=1.8, ls=":", label="contracted position")
    ax.axhline(limit, color=C["limit"], ls="--", lw=1.2)
    ax.axhline(0, color=C["limit"], lw=0.8, alpha=0.5)
    ax.set_ylabel("power [MW]")
    ax.legend(loc="upper left", ncols=2)

    ax = axes[2]
    ax.bar(df.index, df.bess, width=0.9 / 96, color=C["bess"], label="BESS power (+discharge)")
    ax.axhline(0, color=C["limit"], lw=0.8, alpha=0.5)
    ax.set_ylabel("BESS power [MW]")
    ax.legend(loc="upper left")

    ax = axes[3]
    ax.plot(df.index, df.soc, color=C["bess"], lw=1.8, label="state of charge")
    ax.set_ylim(0, 1)
    ax.set_ylabel("SoC [-]")
    ax.legend(loc="upper left")

    ax = axes[4]
    ax.step(df.index, df.rebap, color=C["imbalance"], lw=1.6, where="post", label="reBAP")
    daa = sim.store.daa_prices()["price_eur_per_mwh"]
    daa_day = daa.reindex(df.index.tz_convert("UTC"), method="ffill")
    ax.step(
        df.index,
        daa_day.to_numpy(),
        color=C["price"],
        lw=1.6,
        where="post",
        label="day-ahead price",
    )
    ax.axhline(0, color=C["limit"], lw=0.8, alpha=0.5)
    ax.set_ylabel("price [EUR/MWh]")
    ax.set_xlabel("local time (Europe/Berlin)")
    ax.legend(loc="upper left", ncols=2)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_positions_day(sim: Simulator, out: Path) -> Path:
    """Cumulative contracted position by market vs. physical delivery."""
    starts = sorted(sim.dispatch_records)
    idx = pd.DatetimeIndex(starts).tz_convert("Europe/Berlin")
    markets = ("daa", "ida1", "ida2", "ida3", "idc")
    colors = [C["price"], C["wind"], C["pv"], C["position"], C["bess"]]
    cumulative = np.zeros(len(starts))
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for market, color in zip(markets, colors, strict=True):
        step = np.array([sim.book.net_position_mw_at(s, market) for s in starts])
        cumulative = cumulative + step
        ax.step(idx, cumulative, where="post", lw=1.6, color=color, label=f"after {market.upper()}")
    delivered = np.array([sim.dispatch_records[s].dispatch.grid_power_mw for s in starts])
    ax.step(
        idx, delivered, where="post", lw=2.0, color=C["grid"], ls="--", label="physical delivery"
    )
    ax.fill_between(
        idx, cumulative, delivered, color=C["imbalance"], alpha=0.25, step="post", label="deviation"
    )
    ax.set_ylabel("power [MW]")
    ax.set_xlabel("local time (Europe/Berlin)")
    ax.legend(loc="upper left", ncols=3)
    ax.set_title(f"Contracted position build-up vs. delivery — {idx[0].date()}")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_congestion_day(sim: Simulator, out: Path) -> Path:
    """Oversizing view: unconstrained vs. actual export, absorbed and curtailed."""
    df = _episode_frame(sim)
    limit = sim.cfg.site.grid.export_limit_mw
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)

    ax = axes[0]
    ax.plot(
        df.index,
        df.grid_unconstrained,
        color=C["position"],
        lw=1.6,
        label="unconstrained export (requested)",
    )
    ax.plot(df.index, df.grid, color=C["grid"], lw=1.8, label="actual export")
    ax.axhline(limit, color=C["limit"], ls="--", lw=1.2, label=f"grid limit {limit:.0f} MW")
    ax.fill_between(
        df.index,
        df.grid,
        df.grid_unconstrained,
        where=df.grid_unconstrained > df.grid,
        color=C["curtail"],
        alpha=0.3,
        label="excess resolved",
    )
    ax.set_ylim(top=max(limit, df.grid_unconstrained.max()) * 1.25)
    ax.set_ylabel("power [MW]")
    ax.legend(loc="upper left", ncols=2)
    ax.set_title(f"Grid-connection congestion — {df.index[0].date()}")

    ax = axes[1]
    ax.bar(
        df.index,
        df.congestion_charge,
        width=0.9 / 96,
        color=C["bess"],
        label="BESS charging (congestion)",
    )
    ax.bar(
        df.index,
        -df.congestion_curt,
        width=0.9 / 96,
        color=C["curtail"],
        label="technical curtailment",
    )
    ax.axhline(0, color=C["limit"], lw=0.8, alpha=0.5)
    ax.set_ylabel("power [MW]")
    ax.set_xlabel("local time (Europe/Berlin)")
    ax.legend(loc="upper left", ncols=2)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_export_duration_curve(
    frames: list[pd.DataFrame], labels: list[str], limit_mw: float, out: Path
) -> Path:
    """Grid-export duration curve(s) with the connection limit marked."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    palette = [C["grid"], C["wind"], C["position"], C["bess"]]
    for df, label, color in zip(frames, labels, palette, strict=False):
        sorted_export = np.sort(df["grid"].to_numpy(float))[::-1]
        pct = np.linspace(0, 100, len(sorted_export))
        ax.plot(pct, sorted_export, lw=1.8, color=color, label=label)
    ax.axhline(limit_mw, color=C["limit"], ls="--", lw=1.2, label=f"grid limit {limit_mw:.0f} MW")
    ax.set_xlabel("share of intervals [%]")
    ax.set_ylabel("grid export [MW]")
    ax.legend(loc="upper right")
    ax.set_title("Grid-export duration curve")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_revenue_decomposition(summary: pd.DataFrame, out: Path) -> Path:
    """Stacked revenue decomposition per controller (TOTAL rows)."""
    components = [
        ("daa_revenue_eur", "DAA", C["price"]),
        ("ida1_revenue_eur", "IDA1", C["wind"]),
        ("ida2_revenue_eur", "IDA2", C["pv"]),
        ("ida3_revenue_eur", "IDA3", C["position"]),
        ("idc_revenue_eur", "IDC", C["bess"]),
        ("imbalance_cash_eur", "imbalance", C["imbalance"]),
        ("transaction_cost_eur", "transaction", C["limit"]),
        ("degradation_cost_eur", "degradation", C["curtail"]),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(summary.index))
    pos_base = np.zeros(len(x))
    neg_base = np.zeros(len(x))
    for col, label, color in components:
        values = summary[col].to_numpy(float) / 1e3
        pos = np.clip(values, 0, None)
        neg = np.clip(values, None, 0)
        ax.bar(
            x,
            pos,
            bottom=pos_base,
            color=color,
            width=0.6,
            label=label,
            edgecolor="white",
            linewidth=1.0,
        )
        ax.bar(x, neg, bottom=neg_base, color=color, width=0.6, edgecolor="white", linewidth=1.0)
        pos_base += pos
        neg_base += neg
    totals = summary["total_net_revenue_eur"].to_numpy(float) / 1e3
    for xi, total in zip(x, totals, strict=True):
        ax.plot([xi - 0.34, xi + 0.34], [total, total], color="black", lw=1.6)
        ax.annotate(
            f"{total:,.0f}",
            (xi, total),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=8,
        )
    ax.set_xticks(x, summary.index)
    ax.set_ylabel("cash flow [kEUR]")
    ax.axhline(0, color=C["limit"], lw=0.8)
    ax.set_ylim(bottom=min(0.0, neg_base.min()) * 1.3, top=pos_base.max() * 1.35)
    ax.legend(loc="upper left", ncols=4)
    ax.set_title("Revenue decomposition (black line = net total)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_episode_report(sim: Simulator, out_dir: Path) -> list[Path]:
    """All per-day diagnostic figures for the current episode."""
    day = str(sorted(sim.dispatch_records)[0].tz_convert("Europe/Berlin").date())
    out_dir = Path(out_dir)
    return [
        plot_dispatch_day(sim, out_dir / f"dispatch_{day}.png"),
        plot_positions_day(sim, out_dir / f"positions_{day}.png"),
        plot_congestion_day(sim, out_dir / f"congestion_{day}.png"),
    ]


def episode_frame(sim: Simulator) -> pd.DataFrame:
    """Public accessor for the per-interval episode frame (used by reports)."""
    return _episode_frame(sim)
