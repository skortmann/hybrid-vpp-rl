# Ensemble analysis (Phase 2)

Members: the five eval-best checkpoints (validation means 54,305 – 55,110, std 372 EUR/day). Rule-based reference: 55,511 EUR/day. 92 validation days, 6 blocks.

## Ensembles versus members (blocked validation)

|                       |   mean |   median |   fold_std |   worst_fold |   cvar_regret |   mean_regret |   p_beat_reference |   downside_exposure |
|:----------------------|-------:|---------:|-----------:|-------------:|--------------:|--------------:|-------------------:|--------------------:|
| ensemble_mean         |  55635 |    56916 |       9024 |        43893 |         -1822 |           124 |               0.52 |                 309 |
| ensemble_trimmed_mean |  55622 |    57022 |       9038 |        43719 |         -3869 |           111 |               0.57 |                 618 |
| ensemble_weighted     |  55579 |    56859 |       9023 |        43974 |         -4428 |            68 |               0.6  |                 648 |
| ensemble_median       |  55374 |    56692 |       8872 |        43794 |         -6524 |          -137 |               0.46 |                1068 |
| ckpt_seed2_best       |  55110 |    55486 |       8356 |        43015 |         -9681 |          -401 |               0.51 |                1590 |
| ckpt_seed0_best       |  55067 |    53760 |      10696 |        43386 |        -14788 |          -445 |               0.51 |                2380 |
| ckpt_seed1_best       |  54780 |    56086 |       8458 |        42511 |         -6374 |          -731 |               0.47 |                1151 |
| ckpt_seed4_best       |  54396 |    54083 |       8335 |        43342 |         -9943 |         -1115 |               0.36 |                1948 |
| ckpt_seed3_best       |  54305 |    55799 |       8257 |        42470 |         -8720 |         -1206 |               0.36 |                1696 |

An ensemble is one deterministic controller: it removes the seed-
selection step entirely. Compare its row against the spread of the
member rows to judge the variance reduction.

## Does disagreement predict poor performance?

* Spearman(u, ensemble daily regret): **+0.11**
* Spearman(u, member-mean daily regret): +0.09
* Spearman(u, absolute deviation): -0.07
* mean regret on high-disagreement days (above median u): +262 EUR/day
* mean regret on low-disagreement days: -14 EUR/day
* P(negative regret | high u): 39%; P(negative regret | low u): 57%
