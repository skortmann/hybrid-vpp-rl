"""Canonical machine-readable result summary.

Assembles ``results/final_results.json`` and ``results/final_results.csv``
from the committed result artifacts only — no model rollouts, no private
data — so the published numbers can be regenerated with one command:

    uv run python -m hybrid_vpp.evaluation.export_results

Sources: ``artifacts/robust_selection/per_day_val.csv`` (per-day revenue,
92 validation days), ``artifacts/test_evaluation.json`` (frozen first-phase
test evaluation, 98 days), and
``artifacts/robust_selection/final_test_confirmation.json`` (pre-registered
one-shot confirmation of the promoted controller). Paired statistics use
the moving-block bootstrap of
:mod:`hybrid_vpp.evaluation.hierarchical_bootstrap`.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------- CONFIG

PER_DAY_VAL = Path("artifacts/robust_selection/per_day_val.csv")
TEST_EVALUATION = Path("artifacts/test_evaluation.json")
CONFIRMATION = Path("artifacts/robust_selection/final_test_confirmation.json")
OUT_JSON = Path("results/final_results.json")
OUT_CSV = Path("results/final_results.csv")
PROVENANCE_TAG = "robust-rl-final"
N_BOOT = 5_000
BLOCK_LEN = 7

VALIDATION_CANDIDATES = (
    "ensemble_mean",
    "gate_c_r0.1",
    "gate_a_q80",
    "ckpt_seed0_best",
    "ckpt_seed1_best",
    "ckpt_seed2_best",
    "ckpt_seed3_best",
    "ckpt_seed4_best",
    "baseline_rule_based",
    "baseline_milp_info",
    "baseline_do_nothing",
)

# ------------------------------------------------------------------------


def _row(
    controller: str,
    split: str,
    period: str,
    series: pd.Series,
    reference: pd.Series | None,
    milp: pd.Series | None,
) -> dict:
    from hybrid_vpp.evaluation.hierarchical_bootstrap import hierarchical_bootstrap

    row = {
        "controller": controller,
        "split": split,
        "period": period,
        "n_days": int(series.notna().sum()),
        "mean_eur_per_day": round(float(series.mean()), 2),
        "median_eur_per_day": round(float(series.median()), 2),
        "paired_mean_vs_rule_based": None,
        "paired_ci95_low": None,
        "paired_ci95_high": None,
        "p_outperform_rule_based": None,
        "median_info_gap_pct": None,
        "provenance_tag": PROVENANCE_TAG,
    }
    if reference is not None and controller != "baseline_rule_based":
        boot = hierarchical_bootstrap(
            series.to_frame("x"), reference, n_boot=N_BOOT, block_len=BLOCK_LEN, seed=0
        )
        row.update(
            paired_mean_vs_rule_based=round(boot["mean_diff"], 2),
            paired_ci95_low=round(boot["ci95_low"], 2),
            paired_ci95_high=round(boot["ci95_high"], 2),
            p_outperform_rule_based=round(boot["p_outperform"], 4),
        )
    if milp is not None:
        gap = (milp - series) / milp.abs()
        row["median_info_gap_pct"] = round(float(gap.median() * 100), 3)
    return row


def build_rows() -> list[dict]:
    rows = []

    val = pd.read_csv(PER_DAY_VAL, index_col=0, parse_dates=True)
    val_ref = val["baseline_rule_based"]
    val_milp = val["baseline_milp_info"]
    period = f"{val.index[0].date()}..{val.index[-1].date()}"
    for cand in VALIDATION_CANDIDATES:
        if cand in val.columns:
            rows.append(_row(cand, "validation", period, val[cand], val_ref, val_milp))

    test = json.loads(TEST_EVALUATION.read_text())
    days = pd.DatetimeIndex(pd.Timestamp(d) for d in test["days"])
    t_period = f"{days[0].date()}..{days[-1].date()}"
    baselines = {k: pd.Series(v, index=days) for k, v in test["baselines"].items()}
    t_ref, t_milp = baselines["rule_based"], baselines["milp_info"]
    for name, series in baselines.items():
        rows.append(_row(f"baseline_{name}", "test_reused", t_period, series, t_ref, t_milp))
    seeds = {k: pd.Series(v, index=days) for k, v in test["rl_seeds"].items()}
    for name, series in seeds.items():
        rows.append(_row(f"sac_hybrid_{name}_best", "test_reused", t_period, series, t_ref, t_milp))
    pooled = pd.concat(seeds.values())
    rows.append(
        {
            **_row("sac_hybrid_pooled_5_seeds", "test_reused", t_period, pooled, None, None),
            "median_info_gap_pct": round(
                float(
                    pd.concat([(t_milp - s) / t_milp.abs() for s in seeds.values()]).median() * 100
                ),
                3,
            ),
        }
    )

    conf = json.loads(CONFIRMATION.read_text())
    promoted = pd.Series(
        {pd.Timestamp(d): v for d, v in conf["per_day"].items()}
    ).sort_index()
    rows.append(_row("ensemble_deployment_gate_c_r0.1", "test_reused", t_period, promoted, t_ref, t_milp))
    return rows


def main() -> None:
    rows = build_rows()
    OUT_JSON.parent.mkdir(exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(
            {
                "description": "Validated headline results of the hybrid VPP study; "
                "regenerated from committed artifacts by "
                "hybrid_vpp.evaluation.export_results",
                "economics": "env-v2 (historical reBAP + 25 EUR/MWh deviation penalty)",
                "rows": rows,
            },
            indent=2,
        )
    )
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUT_JSON} and {OUT_CSV} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
