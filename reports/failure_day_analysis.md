# Failure-day analysis (Phase 4)

Split: val (92 days). Mechanism tags are feature quartiles over
the split; lift > 1 means the mechanism is over-represented on the
candidate's failure days.

## ensemble_mean

Worst-regret days: 2025-12-08, 2025-11-02, 2025-12-09, 2025-11-25, 2025-11-05.
Worst revenue decile: 2026-01-01, 2025-11-08, 2025-11-09, 2025-12-05, 2025-11-10, 2025-12-20, 2025-12-03, 2025-11-07, 2025-12-12, 2026-01-08.
Largest deviation: 2026-01-12, 2025-11-04, 2026-01-25, 2025-12-16, 2026-01-26.

Mechanism prevalence on failure days (lift vs all-day base rate):

|                  |   failure_days |   base_rate |   lift |
|:-----------------|---------------:|------------:|-------:|
| rebap_volatility |           0.3  |        0.25 |    1.2 |
| forecast_error   |           0.25 |        0.25 |    1   |
| negative_prices  |           0.05 |        0.01 |    4.6 |
| price_volatility |           0.4  |        0.25 |    1.6 |
| low_renewables   |           0.55 |        0.25 |    2.2 |

## ckpt_seed2_050k

Worst-regret days: 2025-11-12, 2025-12-09, 2025-12-06, 2025-11-20, 2025-12-25.
Worst revenue decile: 2025-11-08, 2025-11-09, 2025-12-05, 2025-11-10, 2025-12-20, 2025-11-07, 2025-12-03, 2026-01-01, 2025-12-12, 2026-01-08.
Largest deviation: 2026-01-01, 2025-11-12, 2026-01-12, 2025-11-01, 2025-11-19.

Mechanism prevalence on failure days (lift vs all-day base rate):

|                  |   failure_days |   base_rate |   lift |
|:-----------------|---------------:|------------:|-------:|
| rebap_volatility |           0.33 |        0.25 |   1.33 |
| forecast_error   |           0.28 |        0.25 |   1.11 |
| negative_prices  |           0.06 |        0.01 |   5.11 |
| price_volatility |           0.33 |        0.25 |   1.33 |
| low_renewables   |           0.44 |        0.25 |   1.78 |

## ckpt_seed2_best

Worst-regret days: 2026-01-05, 2026-01-08, 2025-11-25, 2026-01-02, 2026-01-12.
Worst revenue decile: 2026-01-08, 2025-11-08, 2025-11-09, 2025-12-05, 2026-01-05, 2025-11-10, 2025-12-20, 2025-12-03, 2026-01-01, 2025-12-12.
Largest deviation: 2026-01-01, 2026-01-12, 2025-11-04, 2025-12-02, 2026-01-30.

Mechanism prevalence on failure days (lift vs all-day base rate):

|                  |   failure_days |   base_rate |   lift |
|:-----------------|---------------:|------------:|-------:|
| rebap_volatility |           0.44 |        0.25 |   1.75 |
| forecast_error   |           0.31 |        0.25 |   1.25 |
| negative_prices  |           0.06 |        0.01 |   5.75 |
| price_volatility |           0.5  |        0.25 |   2    |
| low_renewables   |           0.56 |        0.25 |   2.25 |
