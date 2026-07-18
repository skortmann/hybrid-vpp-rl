"""Failure-day identification, market-regime features, mechanism tagging.

For each analysed controller the module identifies its worst validation
days (largest negative regret vs rule-based, worst revenue tail, largest
deviation volume), attaches observable market/regime features to every
day, and tags candidate mechanisms by feature quartiles. The tags are
evidence for Phase 4 decisions — regime features are only added to the
observation space if the dominant mechanisms justify them.

All features are computed from the market database and the environment's
own forecasters. They describe *days* for post-hoc attribution; nothing
here feeds the policies.

Run as ``uv run python -m hybrid_vpp.evaluation.failure_days``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------- CONFIG

SPLIT = "val"
CONFIG_PATH = Path("runs/experiments/V6-hybrid-sac-strategic-seed0.yaml")
CANDIDATES = ("ensemble_mean", "ckpt_seed2_050k", "ckpt_seed2_best")
N_WORST = 5
WORST_FRACTION = 0.10
CACHE_DIR = Path("artifacts/robust_selection/cache")
OUT_DIR = Path("artifacts/failure_days")
REPORT_PATH = Path("reports/failure_day_analysis.md")

# ------------------------------------------------------------------------


def day_features(split: str = SPLIT) -> pd.DataFrame:
    """Observable per-day market and forecast features for attribution."""
    from hybrid_vpp.config.models import load_config
    from hybrid_vpp.core.timegrid import local_day_bounds_utc
    from hybrid_vpp.envs.hybrid_vpp_env import HybridVppEnv
    from hybrid_vpp.evaluation.run_baselines import build_stack

    cfg = load_config(CONFIG_PATH)
    days = HybridVppEnv(cfg, split=split).valid_days
    store, profiles, _sim, renewable_fc, _price_fc = build_stack(cfg)
    daa = store.daa_prices()["price_eur_per_mwh"]
    rebap = store.rebap()

    rows = {}
    for day in days:
        w0, w1 = local_day_bounds_utc(day)
        daa_day = daa.loc[w0 : w1 - pd.Timedelta(minutes=1)]
        rebap_day = rebap.loc[w0 : w1 - pd.Timedelta(minutes=1)]
        realized = profiles.loc[w0 : w1 - pd.Timedelta(minutes=1)]
        realized_mw = realized["wind_avail_mw"] + realized["pv_avail_mw"]
        # day-ahead renewable forecast, issued at the DAA gate the day before
        issue = w0 - pd.Timedelta(hours=12)
        fc = renewable_fc.forecast(issue, realized_mw.index)
        fc_mw = fc["wind_mw"] + fc["pv_mw"]
        err = (fc_mw - realized_mw).abs()
        rows[pd.Timestamp(day)] = {
            "daa_mean": float(daa_day.mean()),
            "daa_std": float(daa_day.std(ddof=0)),
            "negative_price_hours": float((daa_day < 0).mean() * 24) if len(daa_day) else 0.0,
            "rebap_std": float(rebap_day.std(ddof=0)),
            "rebap_abs_max": float(rebap_day.abs().max()),
            "renewable_mwh": float(realized_mw.sum() * 0.25),
            "forecast_mae_mw": float(err.mean()),
            "forecast_maxerr_mw": float(err.max()),
        }
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


MECHANISMS = {
    "rebap_volatility": lambda f, q: f["rebap_std"] >= q["rebap_std"],
    "forecast_error": lambda f, q: f["forecast_mae_mw"] >= q["forecast_mae_mw"],
    "negative_prices": lambda f, q: f["negative_price_hours"] > 0,
    "price_volatility": lambda f, q: f["daa_std"] >= q["daa_std"],
    "low_renewables": lambda f, q: f["renewable_mwh"] <= q["renewable_mwh_low"],
}


def tag_mechanisms(features: pd.DataFrame) -> pd.DataFrame:
    """Boolean mechanism tags per day (upper/lower quartile thresholds)."""
    quartiles = {
        "rebap_std": features["rebap_std"].quantile(0.75),
        "forecast_mae_mw": features["forecast_mae_mw"].quantile(0.75),
        "daa_std": features["daa_std"].quantile(0.75),
        "renewable_mwh_low": features["renewable_mwh"].quantile(0.25),
    }
    return pd.DataFrame(
        {name: fn(features, quartiles) for name, fn in MECHANISMS.items()},
        index=features.index,
    )


def identify_failure_days(
    per_day: pd.DataFrame, candidate: str, n_worst: int = N_WORST, fraction: float = WORST_FRACTION
) -> dict[str, list[str]]:
    """Worst-day sets for one candidate column of the per-day table."""
    regret = per_day[candidate] - per_day["baseline_rule_based"]
    n_frac = max(1, int(np.ceil(fraction * len(per_day))))
    cache = CACHE_DIR / SPLIT / f"{candidate}.json"
    deviation = pd.Series(
        {
            pd.Timestamp(d): rec["metrics"]["abs_deviation_mwh"]
            for d, rec in json.loads(cache.read_text()).items()
        }
    )
    fmt = lambda idx: [str(pd.Timestamp(d).date()) for d in idx]  # noqa: E731
    return {
        "worst_regret": fmt(regret.nsmallest(n_worst).index),
        "worst_revenue_decile": fmt(per_day[candidate].nsmallest(n_frac).index),
        "largest_deviation": fmt(deviation.nlargest(n_worst).index),
    }


def main() -> None:
    from hybrid_vpp.evaluation.checkpoint_matrix import load_per_day_table

    per_day = load_per_day_table(SPLIT)
    features = day_features(SPLIT)
    tags = tag_mechanisms(features)
    base_rate = tags.mean()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    features.join(tags).to_csv(OUT_DIR / "day_features.csv")

    sections, payload = [], {}
    for candidate in CANDIDATES:
        if candidate not in per_day.columns:
            print(f"skip {candidate}: not in per-day table")
            continue
        sets = identify_failure_days(per_day, candidate)
        worst = pd.DatetimeIndex(sorted({d for days in sets.values() for d in days}))
        enrich = {}
        for day in worst:
            regret = float(per_day.at[day, candidate] - per_day.at[day, "baseline_rule_based"])
            enrich[str(day.date())] = {
                "regret_vs_rule_based": regret,
                "revenue": float(per_day.at[day, candidate]),
                "mechanisms": [m for m, v in tags.loc[day].items() if v],
                **{k: float(v) for k, v in features.loc[day].items()},
            }
        payload[candidate] = {"sets": sets, "days": enrich}
        worst_tags = tags.loc[worst].mean()
        lift = (worst_tags / base_rate.replace(0, np.nan)).round(2)
        sections += [
            f"## {candidate}",
            "",
            f"Worst-regret days: {', '.join(sets['worst_regret'])}.",
            f"Worst revenue decile: {', '.join(sets['worst_revenue_decile'])}.",
            f"Largest deviation: {', '.join(sets['largest_deviation'])}.",
            "",
            "Mechanism prevalence on failure days (lift vs all-day base rate):",
            "",
            pd.DataFrame(
                {"failure_days": worst_tags.round(2), "base_rate": base_rate.round(2), "lift": lift}
            ).to_markdown(),
            "",
        ]

    (OUT_DIR / "failure_days.json").write_text(json.dumps(payload, indent=2))
    lines = [
        "# Failure-day analysis (Phase 4)",
        "",
        f"Split: {SPLIT} (92 days). Mechanism tags are feature quartiles over",
        "the split; lift > 1 means the mechanism is over-represented on the",
        "candidate's failure days.",
        "",
        *sections,
    ]
    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines))
    print(f"wrote {REPORT_PATH} and {OUT_DIR}/failure_days.json")


if __name__ == "__main__":
    main()
