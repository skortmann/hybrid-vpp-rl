# Research summary

How the study was run and what it established. Headline numbers are in
[results](results.md); the full narrative including every negative result
is in `reports/final_study_report.md` (preserved at tag `v0.2.0`). The
internal research diaries are
preserved in history at tags `rl-frontier-v1` and `robust-rl-final`.

## Evaluation discipline

All controllers were evaluated on identical fixed episodes (same days,
profiles, prices, forecasts, constraints, costs) against two references:
the **information-equivalent MILP** (rolling optimization on exactly the
forecasts available to the RL agent) and the **perfect-foresight MILP**
(unattainable upper bound, reported but never a target). Splits are
chronological; model selection used validation data only; each research
phase ended with a single locked test evaluation. Where selection rules
or thresholds were compared, they were fixed before evaluation and scored
by leave-one-block-out on contiguous temporal folds.

## Formulation study

Five schema-versioned action spaces were evaluated (`act-v1` … `act-v5`).
Direct incremental orders (103 dims) fail structurally: market decisions
receive ~1% of the gradient signal, and non-idempotent semantics create a
trade-and-pay-imbalance churn loop. Target-position semantics fix the
churn; the strategic space (`act-v5`: seven economic decision variables
translated through the rule-based structure) proved the most effective —
exploration reduces to a small box that contains the rule-based policy as
an interior point. Algorithm choice (PPO, SAC, TQC, TD3, CrossQ)
mattered far less than action semantics and architecture: under identical
economics on the strategic space, off-policy algorithms landed within
~1–2k EUR/day of each other, while PPO was consistently weakest.

A sixth formulation (`act-v6`, `strategic_residual`) extends the
strategic space after the main study: a low-rank quarter-hour market
residual — one anchor and one zero-mean intra-hour tilt per hour anchor,
57 dims for one-day episodes — restores intra-hour degrees of freedom on
the always-quarter-hourly IDA/IDC products (and day-ahead after the
2025-10-01 MTU switch) and lets the policy trade products the rule-based
baseline leaves untouched, while dispatch stays with the deterministic
tracker. Zero residuals reproduce `act-v5` exactly and mid-range values
reproduce the rule-based controller (both test-pinned); its screening
follows the same fixed-validation-day protocol. See the
[environment page](rl_environment.md) for the full translation semantics.

## Validity findings

Under the original economics, expressive policies exceeded the
perfect-foresight bound — a red flag, not a success. Profit-decomposition
audits showed 50–67% of their profit came from deliberate deviation
settled at historical reBAP with no penalty: **imbalance speculation**,
legal in-model but inadmissible as market operation. The economics were
corrected (25 EUR/MWh deviation penalty for *all* controllers, anchors
recomputed), and the contaminated results were annotated and excluded
from operational rankings rather than deleted.

Two further mechanisms were isolated and fixed by intervention: the
rule-equivalent optimum sat on the action-box corner, unreachable for
squashed-Gaussian policies (fixed by interiorizing the gain range), and
risk-neutral policies carried 3–6× the reference deviation, converting
volatile-reBAP days into five-figure losses (fixed architecturally —
**deterministic rule-based dispatch under RL market control**, the
decisive design change of the study).

## Robustness study

The first phase's residual weakness was selection: the validation-best
seed was the worst test generalizer. The second phase replaced selection
with construction: blocked temporal validation over all saved checkpoints
(seed and checkpoint effects turned out comparable, with early
checkpoints generalizing best), a pre-registered risk-adjusted selection
score (the only rule family with positive held-out regret), mean-action
**ensembling** of the five frozen policies (beats every member, cuts tail
regret several-fold), and a **bounded residual** around the
rule-equivalent action (worst-day losses capped at design time). A
random-fallback control experiment showed that ensemble disagreement is
not a causal safety signal — the safety value comes from bounded
dampening itself. Regime conditioning was pruned for lack of evaluable
evidence (the dominant failure regime occurs on one winter validation
day), and stronger MILP benchmark variants (turnover-penalized,
forecast-derated) confirmed rather than closed the robustness gap.

## Tested and insufficient

All multi-seed, all recorded: direct-action PPO; target-position
formulations under the uncorrected economics; PPO generally under the
corrected economics; behavior-cloned initialization (twice — washes out
during entropy-regularized fine-tuning); replay prefill and risk shaping
(within seed noise); the median-action ensemble; disagreement gating
beyond its random-fallback control. Pruned with stated reasons rather
than disproven: extended training budgets, offline IQL/CQL, model-based
methods (sample efficiency was demonstrably not the binding constraint),
regime-aware retraining, and critic-weighted ensembles (Q-calibration
across seeds not established).

## Outcome

The promoted design — mean strategic action of five frozen SAC policies,
bounded residual 0.1 around the rule-equivalent action, deterministic
dispatch — achieves median-level parity with the information-equivalent
MILP, eliminates seed-selection risk by construction, and tracks the
rule-based reference within ±0.2% of mean revenue with a hard cap on
daily losses. It does **not** demonstrate a mean-revenue advantage over
the rule-based controller on unseen data; see
[limitations](limitations.md).
