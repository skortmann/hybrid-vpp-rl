# Regime analysis (Phase 4)

Evidence base: `reports/failure_day_analysis.md` (mechanism tags over the
92 validation days) and two controlled gate experiments.

## What the failure days say

For all three analysed controllers (mean ensemble, `seed2_050k`,
`seed2_best`) the same mechanisms are over-represented on
worst-regret/worst-decile days:

| mechanism | base rate | lift on failure days |
|---|---|---|
| negative DAA prices | 1% (1 day) | 4.6 – 5.8 |
| low renewable generation (lowest quartile) | 25% | 1.8 – 2.3 |
| DAA price volatility (upper quartile) | 25% | 1.3 – 2.0 |
| reBAP volatility (upper quartile) | 25% | 1.2 – 1.8 |
| day-ahead renewable forecast error | 25% | 1.0 – 1.3 |

reBAP volatility — the dominant failure mechanism of the previous
research phase — is now only mildly enriched: the hybrid architecture
(deterministic rule-based dispatch) already neutralized it. Forecast
error is not enriched at all; the remaining regret is a *market-side*
phenomenon (trading decisions on hard days), not a forecast phenomenon.

## Controlled regime experiments

* **Gate E (negative-price fallback)**: fall back to rule-based control
  on days whose day-ahead price forecast (available at the DAA gate)
  contains negative prices. Result: the trigger fires on **1 of 92
  validation days** — the mechanism with the highest lift is so rare in
  the Nov–Jan validation window that no regime rule about it can be
  calibrated or validated there (mean 55,551, within noise of the
  ungated ensemble). Negative-price days matter in spring (the reused
  test window), which is precisely the period no validation protocol on
  this dataset can reach.
* **Random-fallback control**: rule-based fallback on a seeded-random
  20% of market events recovers most of Gate A's tail improvement
  (CVaR regret −1,441 vs −1,032 for disagreement-triggered fallback,
  ungated −1,822). Together with the weak day-level correlation
  (Spearman +0.11), this shows the safety value comes from *bounded
  dampening toward the rule action*, not from the disagreement signal.

## Decision

No regime features are added to the observation schema in this phase:

1. The only strongly enriched, observable regime (negative prices) has a
   1% base rate in the validation period — any fitted rule would be
   unvalidatable (n=1) and an invitation to overfit.
2. The bounded-residual gate already achieves the downside protection a
   regime fallback would target (downside exposure 77 EUR/day, max
   daily loss 1,430), without needing to know *which* regime it is in.
3. Low-renewable and price-volatility enrichment is expressed through
   many small losses, not catastrophic days; the bounded residual caps
   these by construction.

Regime conditioning (Methods 1–4 of the plan) is therefore **formally
pruned** for this phase, with negative-price handling recorded as the
first candidate mechanism for a future phase with spring/summer
validation data.
