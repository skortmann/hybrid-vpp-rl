"""One-week market-stage dispatch figure for a trained RL checkpoint.

Rolls the checkpoint deterministically over one week of the chosen split
and renders a single four-panel figure: delivered wind/PV production,
net contracted position per market stage (DAA, IDA1-3, IDC), the summed
position against the delivered grid export, and the battery SoC — the
trajectory *implied* by the book after each successive market stage
(cumulative position minus delivered production, integrated under the
battery's ratings, efficiencies, and SoC window, reset to the initial SoC
at each local-midnight episode boundary) together with the realized
(final) SoC from the simulator. The checkpoint and its training config
are resolved from ``experiments/registry.jsonl`` by experiment id
(override with explicit paths if the registry is absent).

Run as ``uv run python -m hybrid_vpp.evaluation.market_stage_plot``; edit
the CONFIG block or call :func:`plot_market_stage_week` with keyword
arguments.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from hybrid_vpp.config.models import load_config
from hybrid_vpp.markets.positions import MARKETS

# --------------------------------------------------------------------- CONFIG

EXPERIMENT_ID = "V6-hybrid-sac-strategic-seed0"
MONDAY = "2025-11-24"  # first day of the plotted week (any weekday works)
SPLIT = "val"
#: chain the battery SoC across the week (physically faithful roll-over)
#: instead of resetting to soc_initial at each local-midnight boundary.
#: Off by default to match the published daily-reset evaluation regime;
#: set True to visualize continuous end-of-day carry-over.
CARRY_OVER = False
REGISTRY = Path("experiments/registry.jsonl")
#: set both to bypass the registry lookup
MODEL_PATH: Path | None = None
CONFIG_PATH: Path | None = None
FIG_DIR = Path("reports/figures")
TZ = "Europe/Berlin"

# ------------------------------------------------------------------------

_C_WIND, _C_PV = "#2a78d6", "#008300"
_C_MARKET = {
    "daa": "#e87ba4",
    "ida1": "#eda100",
    "ida2": "#1baf7a",
    "ida3": "#eb6834",
    "idc": "#4a3aa7",
}
_C_EXPORT = "#e34948"
_SURFACE, _INK, _INK2, _MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
_GRIDLINE, _BASELINE = "#e1e0d9", "#c3c2b7"


def _resolve_checkpoint(experiment_id: str) -> tuple[Path, Path]:
    """Latest (model_path, config_path) for the experiment id from the registry."""
    found: tuple[Path, Path] | None = None
    for line in REGISTRY.read_text().splitlines():
        record = json.loads(line)
        if record.get("experiment_id") == experiment_id:
            found = (Path(record["model_path"]), Path(record["config_path"]))
    if found is None:
        raise LookupError(f"experiment {experiment_id!r} not found in {REGISTRY}")
    return found


def _collect_week(model_path: Path, config_path: Path, monday: str, split: str, carry_over: bool):
    """Quarter-hour production, per-market positions, and export for one week.

    With ``carry_over`` the battery SoC is chained day to day (each episode
    starts at the previous day's final SoC); otherwise it resets to
    ``soc_initial`` at every episode as usual.
    """
    from hybrid_vpp.envs.hybrid_vpp_env import HybridVppEnv
    from hybrid_vpp.training.algorithms import algo_class

    cfg = load_config(config_path)
    cfg.episode.initial_soc_range = None  # deterministic figure
    model = algo_class(cfg.training.algorithm).load(model_path)
    env = HybridVppEnv(cfg, split=split)
    rows, revenue, carried = [], 0.0, None
    for day in [str(d.date()) for d in pd.date_range(monday, periods=7)]:
        options = {"day": day}
        if carry_over and carried is not None:
            options["initial_soc"] = carried
        obs, info = env.reset(options=options)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, info = env.step(action)
        revenue += float(info["episode_metrics"]["total_net_revenue_eur"])
        carried = float(info["episode_metrics"]["final_soc"])
        for t, rec in sorted(env.sim.dispatch_records.items()):
            row = {
                "time_utc": t,
                "wind_mw": rec.wind_avail_mw - rec.dispatch.wind_curtail_mw,
                "pv_mw": rec.pv_avail_mw - rec.dispatch.pv_curtail_mw,
                "grid_mw": rec.dispatch.grid_power_mw,
                "bess_mw": rec.dispatch.bess_power_mw,
                "soc": rec.battery.energy_after_mwh / cfg.site.battery.energy_capacity_mwh,
            }
            for market in MARKETS:
                row[market] = env.sim.book.net_position_mw_at(t, market)
            rows.append(row)
    return pd.DataFrame(rows), revenue, cfg.site.battery


def _implied_soc(position: pd.Series, production: pd.Series, bat, carry_over: bool) -> pd.Series:
    """SoC implied by delivering ``position`` against realized production.

    Battery power = position − production, clipped to the ratings; energy is
    integrated with charge/discharge efficiencies inside the SoC window.
    Without ``carry_over`` it resets to the initial SoC at each local-day
    boundary (1-day episodes); with ``carry_over`` it flows continuously.
    """
    cap = bat.energy_capacity_mwh
    e_min, e_max = bat.soc_min * cap, bat.soc_max * cap
    e, day = bat.soc_initial * cap, None
    out = []
    for t in position.index:
        if t.date() != day:
            day = t.date()
            if not carry_over:
                e = bat.soc_initial * cap
        power = min(max(position[t] - production[t], -bat.charge_power_mw), bat.discharge_power_mw)
        if power >= 0:
            e -= min(power * 0.25 / bat.discharge_efficiency, e - e_min)
        else:
            e += min(-power * 0.25 * bat.charge_efficiency, e_max - e)
        out.append(e / cap)
    return pd.Series(out, index=position.index)


def _soc_composition(d: pd.DataFrame, bat, carry_over: bool) -> pd.DataFrame:
    """Stored energy split by origin: initial fill, wind, PV, grid purchase.

    Charging energy is attributed to grid import where the interval imports
    (`grid_mw < 0`), otherwise to the delivered renewables (split by their
    wind/PV shares). Discharge drains all buckets proportionally (perfect
    mixing) and every interval is rescaled to the realized SoC, so the
    stack's top equals the realized SoC. Without ``carry_over`` the buckets
    reset with each daily episode; with ``carry_over`` they flow across days
    (the "initial fill" bucket then only seeds the first day).
    """
    cap = bat.energy_capacity_mwh
    buckets = {"initial": bat.soc_initial * cap, "wind": 0.0, "pv": 0.0, "grid": 0.0}
    day, rows = None, []
    for t, row in d.iterrows():
        if t.date() != day:
            day = t.date()
            if not carry_over:
                buckets = {"initial": bat.soc_initial * cap, "wind": 0.0, "pv": 0.0, "grid": 0.0}
        power = row["bess_mw"]
        if power < 0:  # charging
            charged = -power * 0.25 * bat.charge_efficiency
            from_grid = min(-power, max(0.0, -row["grid_mw"])) * 0.25 * bat.charge_efficiency
            renewable = max(row["wind_mw"] + row["pv_mw"], 1e-9)
            buckets["grid"] += from_grid
            buckets["wind"] += (charged - from_grid) * row["wind_mw"] / renewable
            buckets["pv"] += (charged - from_grid) * row["pv_mw"] / renewable
        elif power > 0:  # discharging drains proportionally
            total = sum(buckets.values())
            drained = min(power * 0.25 / bat.discharge_efficiency, total)
            if total > 1e-9:
                for key in buckets:
                    buckets[key] *= 1.0 - drained / total
        total = sum(buckets.values())
        realized = row["soc"] * cap
        if total > 1e-9:  # pin the stack to the simulator's realized SoC
            for key in buckets:
                buckets[key] *= realized / total
        rows.append({k: v / cap for k, v in buckets.items()})
    return pd.DataFrame(rows, index=d.index)


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(_SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(_BASELINE)
    ax.grid(True, axis="y", color=_GRIDLINE, linewidth=0.6)
    ax.tick_params(colors=_MUTED, labelsize=8, length=0)


def _label_end(ax, x, y, text, color, dy=0) -> None:
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(4, dy),
        textcoords="offset points",
        fontsize=8,
        color=color,
        va="center",
        fontweight="bold",
    )


def plot_market_stage_week(
    experiment_id: str = EXPERIMENT_ID,
    monday: str = MONDAY,
    split: str = SPLIT,
    model_path: Path | None = MODEL_PATH,
    config_path: Path | None = CONFIG_PATH,
    fig_dir: Path = FIG_DIR,
    carry_over: bool = CARRY_OVER,
) -> Path:
    """Render the weekly market-stage figure and return the written path."""
    if model_path is None or config_path is None:
        model_path, config_path = _resolve_checkpoint(experiment_id)
    df, revenue, battery = _collect_week(model_path, config_path, monday, split, carry_over)
    d = df.set_index("time_utc").sort_index()
    d.index = d.index.tz_convert(TZ)
    t0 = pd.Timestamp(monday, tz=TZ)
    t1 = t0 + pd.Timedelta(days=7)
    d = d.loc[t0:t1]

    fig, axes = plt.subplots(
        5,
        1,
        figsize=(13.5, 13.0),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.2, 1.0, 1.0, 1.0]},
    )
    fig.patch.set_facecolor(_SURFACE)

    ax = axes[0]  # delivered production, stacked
    _style(ax)
    ax.stackplot(d.index, d["wind_mw"], d["pv_mw"], colors=[_C_WIND, _C_PV], alpha=0.45)
    ax.plot(d.index, d["wind_mw"], color=_C_WIND, linewidth=1.0)
    ax.plot(d.index, d["wind_mw"] + d["pv_mw"], color=_C_PV, linewidth=1.0)
    ax.set_ylabel("production MW", fontsize=8, color=_MUTED)
    soc_mode = "SoC carried across days" if carry_over else "SoC resets daily"
    ax.set_title(
        f"{experiment_id} — production and market-stage positions, "
        f"week {monday} ({split}, {soc_mode})",
        fontsize=12,
        color=_INK,
        loc="left",
        pad=10,
    )
    ax.text(
        1.0,
        1.05,
        f"Σ net revenue {revenue / 1000:,.0f}k €",
        transform=ax.transAxes,
        fontsize=9.5,
        color=_INK2,
        ha="right",
    )
    _label_end(ax, d.index[-1], float(d["wind_mw"].iloc[-8]), "wind", _C_WIND)
    _label_end(ax, d.index[-1], float((d["wind_mw"] + d["pv_mw"]).iloc[-8]) + 8, "+ PV", _C_PV)

    ax = axes[1]  # net contracted position per market stage
    _style(ax)
    ax.axhline(0, color=_BASELINE, linewidth=0.8)
    for market in MARKETS:
        width = 1.8 if market == "daa" else 1.2
        ax.step(d.index, d[market], where="post", color=_C_MARKET[market], linewidth=width)
    ax.set_ylabel("position MW", fontsize=8, color=_MUTED)
    ends = sorted([(m, float(d[m].iloc[-8])) for m in MARKETS], key=lambda kv: kv[1])
    min_gap = 0.11 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    ys: list[float] = []
    for _, value in ends:
        ys.append(value if not ys else max(value, ys[-1] + min_gap))
    for (market, _), y in zip(ends, ys, strict=True):
        _label_end(ax, d.index[-1], y, market.upper(), _C_MARKET[market])

    ax = axes[2]  # summed position vs delivered export
    _style(ax)
    total = d[list(MARKETS)].sum(axis=1)
    ax.plot(d.index, total, color=_INK2, linewidth=1.6)
    ax.plot(d.index, d["grid_mw"], color=_C_EXPORT, linewidth=1.3)
    ax.axhline(0, color=_BASELINE, linewidth=0.8)
    ax.set_ylabel("MW", fontsize=8, color=_MUTED)
    _label_end(ax, d.index[-1], float(total.iloc[-8]) + 10, "net position", _INK2)
    _label_end(ax, d.index[-1], float(d["grid_mw"].iloc[-8]) - 10, "delivered export", _C_EXPORT)

    ax = axes[3]  # SoC implied after each stage + realized (final)
    _style(ax)
    production = d["wind_mw"] + d["pv_mw"]
    cumulative = pd.Series(0.0, index=d.index)
    for market in MARKETS:
        cumulative = cumulative + d[market]
        soc = _implied_soc(cumulative, production, battery, carry_over)
        ax.plot(d.index, soc * 100, color=_C_MARKET[market], linewidth=1.1, alpha=0.9)
    ax.plot(d.index, d["soc"] * 100, color=_INK, linewidth=1.8)
    for bound in (battery.soc_min, battery.soc_max):
        ax.axhline(bound * 100, color=_BASELINE, linewidth=0.8, linestyle=":")
    ax.set_ylabel("SoC %", fontsize=8, color=_MUTED)
    ax.set_ylim(0, 100)
    _label_end(ax, d.index[-1], float(d["soc"].iloc[-8]) * 100, "final (realized)", _INK, dy=8)
    _label_end(ax, d.index[-1], float(soc.iloc[-8]) * 100, "after IDC", _C_MARKET["idc"], dy=-8)

    ax = axes[4]  # stored energy by origin (stack top = realized SoC)
    _style(ax)
    comp = _soc_composition(d, battery, carry_over)
    ax.stackplot(
        d.index,
        comp["initial"] * 100,
        comp["wind"] * 100,
        comp["pv"] * 100,
        comp["grid"] * 100,
        colors=[_MUTED, _C_WIND, _C_PV, _C_EXPORT],
        alpha=0.55,
        linewidth=0,
    )
    ax.plot(d.index, d["soc"] * 100, color=_INK, linewidth=1.2)
    for bound in (battery.soc_min, battery.soc_max):
        ax.axhline(bound * 100, color=_BASELINE, linewidth=0.8, linestyle=":")
    ax.set_ylabel("SoC by origin %", fontsize=8, color=_MUTED)
    ax.set_ylim(0, 100)
    layers = comp.iloc[-8] * 100
    y_mid = {
        "initial": layers["initial"] / 2,
        "wind": layers["initial"] + layers["wind"] / 2,
        "pv": layers["initial"] + layers["wind"] + layers["pv"] / 2,
        "grid": layers.sum() - layers["grid"] / 2,
    }
    for key, color, dy in [
        ("initial", _MUTED, -12),
        ("wind", _C_WIND, -4),
        ("pv", _C_PV, 4),
        ("grid", _C_EXPORT, 12),
    ]:
        _label_end(ax, d.index[-1], y_mid[key], f"from {key}", color, dy=dy)
    ax.xaxis.set_major_locator(mdates.DayLocator(tz=d.index.tz))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b", tz=d.index.tz))
    ax.set_xlim(t0, t1)

    handles = [
        plt.Line2D([], [], color=color, linewidth=2, label=label)
        for label, color in [
            ("wind production", _C_WIND),
            ("PV production", _C_PV),
            ("DAA", _C_MARKET["daa"]),
            ("IDA1", _C_MARKET["ida1"]),
            ("IDA2", _C_MARKET["ida2"]),
            ("IDA3", _C_MARKET["ida3"]),
            ("IDC", _C_MARKET["idc"]),
            ("net position", _INK2),
            ("delivered export", _C_EXPORT),
            ("realized SoC (final)", _INK),
            ("stored from grid purchase", _C_EXPORT),
            ("stored from initial fill", _MUTED),
        ]
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=8.5,
        bbox_to_anchor=(0.5, -0.002),
        labelcolor=_INK2,
    )
    fig.align_ylabels(axes)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_carryover" if carry_over else ""
    out = fig_dir / f"market_stages_{experiment_id}_{monday}{suffix}.png"
    fig.savefig(out, dpi=150, facecolor=_SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    plot_market_stage_week()
