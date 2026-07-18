# Results

Validated findings of the study, under penalized-imbalance economics
(historical reBAP settlement plus a 25 EUR/MWh deviation penalty, applied
identically to every controller). Every number states its statistic
(mean/median), population (single seed, pooled seeds, ensemble), and data
split. The machine-readable version is `results/final_results.{json,csv}`,
regenerated from committed artifacts by:

```bash
uv run python -m hybrid_vpp.evaluation.export_results
```

The complete narrative, including all negative results, is in
[`reports/final_study_report.md`](https://github.com/skortmann/hybrid-vpp-rl/blob/main/reports/final_study_report.md).

## Data splits

Chronological, from the 696-day window in which all five markets, reBAP,
and zone data overlap (2024-06-14 → 2026-05-10): training to 2025-10-31,
validation 2025-11-01 → 2026-01-31 (92 winter days), test 2026-02-01 →
2026-05-09 (98 late-winter/spring days). The test split was evaluated
once per research phase; the second read-out is labelled *reused test* —
it is a pre-registered confirmation, not an untouched-test claim.

## Test split (98 days, reused)

| Controller | Mean EUR/day | Median EUR/day |
|---|---|---|
| **Ensemble deployment controller** (promoted) | **46,767** | **48,517** |
| Rule-based | 46,850 | 48,464 |
| Information-equivalent MILP | 47,599 | 49,294 |
| Perfect-foresight MILP (upper bound) | 52,007 | 52,573 |
| SAC hybrid, pooled 5 seeds | 45,147 | 48,174 |
| SAC hybrid, per-seed mean range | 40,736 – 48,104 | — |
| Do-nothing | 42,512 | 41,462 |

Promoted controller, paired against rule-based (moving-block bootstrap):
mean **−83 EUR/day, CI95 [−155, −30]** — statistically distinguishable
from zero, economically small (≈0.2% of daily revenue); median regret −90;
maximum single-day loss **847 EUR**; CVaR₁₀% of daily regret −569;
median information-equivalent gap **+0.58%** (a median statistic — the
mean gap versus the information-equivalent MILP is ≈1.7%).

The reliability improvement over selecting a single trained seed is the
headline robustness result: the paired confidence-interval width narrows
from 5,752 EUR/day (pooled seeds) to **125 EUR/day**, the worst day
improves from five-figure losses to 847 EUR, and no seed selection exists
anywhere in the deployment path. The previous procedure's
validation-chosen seed landed 6,114 EUR/day below rule-based on the same
test days; the best seed in hindsight (48,104 mean) could not be
identified in advance.

## Validation split (92 days)

| Controller | Mean EUR/day | Median EUR/day | P(outperform rule-based) |
|---|---|---|---|
| Best single checkpoint in hindsight (seed 2, 50k steps) | 56,014 | 56,791 | — |
| Mean-action ensemble (ungated) | 55,635 | 56,916 | 0.81 |
| **Ensemble + bounded residual 0.1** (promoted) | 55,538 | 56,988 | **0.91** |
| Rule-based | 55,511 | 57,001 | — |
| Information-equivalent MILP | 55,260 | 57,892 | — |
| Do-nothing | 52,726 | 55,315 | — |

Full per-day series for all 35 checkpoints, ensembles, gates, and
baselines are in `artifacts/robust_selection/per_day_val.csv`. On
validation the promoted controller's
paired mean versus rule-based is +27 EUR/day (CI95 [−12, +71]); every
ensemble variant beat every individual member on mean revenue, and the
mean ensemble cut the tail risk (CVaR₁₀% of daily regret) from
−6,374…−14,788 EUR (members) to −1,822, with the bounded residual
reducing it further to −437.

### Validation figures

Rule-based, information-equivalent MILP, and the two RL composites on
the same 92 days (regenerate with
`uv run python -m hybrid_vpp.evaluation.robust_plots`):

![Cumulative paired difference vs rule-based](assets/robust/val_cumulative_vs_rule_based.png)

The cumulative view shows the controllers' characters: the MILP swings
between +29 kEUR and −23 kEUR against rule-based — it wins ordinary days
on point-forecast optimization and loses multiples of that on hard days —
while the ungated ensemble ends +11 kEUR with one drawdown episode in
December, and the promoted bounded variant accumulates +2.5 kEUR nearly
monotonically.

![Daily regret distribution vs rule-based](assets/robust/val_regret_distribution.png)

The paired-regret distribution is the promotion argument in one figure:
the MILP's daily regret spans −15.7 k to +25 kEUR, the ensemble
compresses it to roughly ±5 kEUR, and the bounded residual to a band of
a few hundred euros — with the median at or above zero for both RL
composites.

![Day-level pairing vs rule-based](assets/robust/val_scatter_vs_rule_based.png)

![Rolling revenue of all controllers](assets/robust/val_rolling_revenue.png)

## Reading the result

* **Median-level parity with the optimization benchmark**: median
  information-equivalent gaps are +0.07% (pooled SAC seeds) and +0.58%
  (promoted controller) on the reused test, and negative (better than the
  MILP) on validation.
* **No demonstrated mean-revenue advantage over rule-based control**: the
  validation edge (+27 EUR/day, P 0.91) inverted to −83 EUR/day on the
  spring test window. The bounded residual caps both tails; its measured
  insurance premium is ≈0.2% of revenue.
* **Robustness, not expectation, is the contribution**: seed-selection
  risk is eliminated by construction, worst-case daily behavior is a
  design parameter, and the result is exactly reproducible from frozen
  checkpoints.

See [limitations](limitations.md) for what these numbers do not show.
