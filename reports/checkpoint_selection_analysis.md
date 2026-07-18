# Checkpoint-selection analysis (Phase 1)

92 validation days (2025-11-01 → 2026-01-31), 6 contiguous
blocks, 35 candidates (5 seeds x 7 checkpoints). Baselines:
do_nothing 52,726, milp_info 55,260, rule_based 55,511 EUR/day.

## Seed versus checkpoint variance

* seed effect std: **517 EUR/day**
* checkpoint effect std: **314 EUR/day**
* seed x checkpoint interaction std: 403 EUR/day
* mean within-seed spread over checkpoints: 512
* mean within-checkpoint spread over seeds: 639

## Checkpoint age

Mean validation revenue by training progress (EUR/day):

| ckpt   |   mean_revenue |
|:-------|---------------:|
| 050k   |          55130 |
| 100k   |          54239 |
| 150k   |          54760 |
| 200k   |          54731 |
| 250k   |          54524 |
| 300k   |          54254 |
| best   |          54731 |

Best checkpoint per seed (full 92-day validation): seed0: 050k, seed1: 200k, seed2: 050k, seed3: 050k, seed4: 150k.

## Selection-rule reliability (leave-one-block-out)

Mean/worst held-out-block performance of the candidate each rule picks:

| rule               |   ('holdout_mean', 'mean') |   ('holdout_mean', 'min') |   ('holdout_regret_mean', 'mean') |   ('holdout_regret_mean', 'min') |   ('gap_to_oracle', 'mean') |   ('gap_to_oracle', 'min') |
|:-------------------|---------------------------:|--------------------------:|----------------------------------:|---------------------------------:|----------------------------:|---------------------------:|
| highest_mean       |                      54912 |                     41789 |                              -704 |                            -2114 |                        2452 |                       1328 |
| highest_median     |                      54863 |                     43321 |                              -752 |                            -2491 |                        2500 |                        608 |
| highest_worst_fold |                      53347 |                     42219 |                             -2268 |                            -5560 |                        4016 |                       1428 |
| lowest_downside    |                      55970 |                     43922 |                               354 |                            -1506 |                        1394 |                          8 |
| pareto             |                      55616 |                     43321 |                                 1 |                            -1506 |                        1747 |                        282 |
| risk_adjusted      |                      55970 |                     43922 |                               354 |                            -1506 |                        1394 |                          8 |

Locked selection rule (highest mean held-out revenue): **risk_adjusted**.

## Do action statistics predict validation regret?

Spearman correlation of mean translated parameters with mean regret:

|               |   spearman_vs_regret |
|:--------------|---------------------:|
| daa_coverage  |                -0.71 |
| daa_arb_scale |                -0.6  |
| ida_gain      |                 0.71 |
| idc_gain      |                 0.69 |

## Read-only test read-out (previous phase, five eval-best checkpoints)

Recorded per-day test results from `artifacts/test_evaluation.json`;
computed after the LOBO ranking above and never used for selection:

|                 |   mean |   median |   selection_score |   test_mean |
|:----------------|-------:|---------:|------------------:|------------:|
| ckpt_seed0_best |  55067 |    53760 |             42213 |       40736 |
| ckpt_seed1_best |  54780 |    56086 |             47181 |       46975 |
| ckpt_seed2_best |  55110 |    55486 |             45991 |       48104 |
| ckpt_seed3_best |  54305 |    55799 |             45514 |       43783 |
| ckpt_seed4_best |  54396 |    54083 |             44978 |       46139 |

Spearman(validation mean, test mean) over the five best checkpoints: **+0.40**; Spearman(risk-adjusted selection score, test mean): **+0.80**. The 30-day window used by the previous phase ranked seed 0 first — the worst test generalizer; the blocked risk-adjusted score ranks it last.
