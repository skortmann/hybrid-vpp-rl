# Limitations

What the published results do and do not support. The claims in
[results](results.md) hold only within these boundaries.

## No demonstrated expected-profit advantage over rule-based control

The promoted controller's paired mean versus the rule-based reference on
the reused 98-day test split is −83 EUR/day (CI95 [−155, −30]) —
statistically distinguishable from zero on the negative side, economically
small (≈0.2% of daily revenue). Median-level parity and a large reduction
in tail risk and selection variance are demonstrated; a higher expected
revenue is not. Deployments whose only objective is expected revenue have
no evidence-based reason to replace a well-tuned rule-based controller
with this design.

## Seasonal transfer is unverified

The validation period is winter-only (November–January); the test period
is late winter to spring. The +27 EUR/day validation edge inverted to
−83 EUR/day across that boundary, and no protocol on this dataset can
test summer behavior: the study window ends 2026-05-10. Negative-price
periods — the mechanism most over-represented on the controller's worst
days (lift 4.6–5.8) — occur on a single validation day and concentrate in
exactly the seasons the data does not cover.

## The test split is reused

The 98-day test split was evaluated once by each research phase. The
second evaluation followed a protocol pre-registered before any test
contact (locked candidate, one deterministic pass, statistics fixed in
advance), but it is a *reused-test confirmation*, not an untouched-test
claim. A genuinely untouched claim requires post-May-2026 data.

## Economic model boundaries

* **Price-taker execution everywhere.** Auctions fill at historical
  clearing prices; continuous intraday fills at published ID1/ID3/IDFULL
  indices by remaining lead time, with volume caps and transaction costs.
  There is no order book, no partial fill, no market impact.
* **Compliance proxy.** The 25 EUR/MWh deviation penalty represents
  balancing-group discipline. Results are calibrated to that regime: in a
  zero-shot sweep the controller ordering is stable from 0 to 50 EUR/MWh,
  but at 100 EUR/MWh the low-deviation information-equivalent MILP
  overtakes all trading-heavier controllers.
* **reBAP settlement is single-price and historical**; the price-taker
  assumption also applies to imbalance volumes.
* **Daily SoC reset.** Episodes are single days and the battery restarts
  at 50% SoC, so the raw ledger metric leaves end-of-day stored energy
  unpriced — a free refill worth about +0.8k EUR/day for the promoted-seed
  RL controller and +1.0k EUR/day for rule-based on validation, while the
  MILP alone was terminal-constrained and received no such subsidy. The
  headline tables use the raw metric (legacy); a terminal-adjusted revenue
  (`total_net_revenue_terminal_adjusted_eur`, same boundary valuation the
  training reward already used) is now reported alongside it, and a
  symmetric unconstrained MILP variant (`milp_no_terminal`) exists for
  like-for-like comparison. The adjustment shifts RL/rule-based revenues
  down by roughly 2% and does not change the promoted design. True
  carry-over of the battery state across days is available as an opt-in
  (`episode.carry_over_soc`, plus a chained-horizon evaluator); the
  published results predate it and use the daily-reset protocol.

## Data and forecast simplifications

* Site profiles are zone-scaled ENTSO-E actuals by default (smoother than
  true single-site output); Renewables.ninja profiles are supported but
  optional.
* Site-level forecasts are synthetic: a leakage-safe error model fitted on
  the training partition of zone forecast errors. Real site forecast
  quality will differ.
* IDA2/IDA3 have 13/10 missing auction days in the window, handled as
  "auction not held".

## Requirements for live deployment

The deployment controller assumes: the frozen released checkpoints (or a
retrained, re-validated ensemble), the rule-based controller available as
the fallback and residual anchor, deterministic dispatch retained, a
calibration of observation plausibility bounds for the OOD fallback, and
monitoring of the decision log. Live trading additionally requires
everything the economic model excludes: order execution, credit and
collateral, balancing-group compliance processes, and regulatory
requirements — none of which are modeled here.
