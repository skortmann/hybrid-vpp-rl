# The hybrid VPP reinforcement-learning study — complete summary

Compiled 2026-07-18 from the repository's validated artifacts: the two final
reports (`final_rl_results.*`, `final_robust_rl_results.*`), the phase reports
under `reports/`, the research registry (`runs/research_state.sqlite`: 49
completed, 5 promoted, 2 pruned; both loops in named terminal states),
`experiments/registry.jsonl` (48 records), the research logs, and the raw
evaluation caches under `artifacts/`. Verified state: branch
`research/robust-rl-selection` at `f2d467c`, tag `rl-frontier-v1` at `17b002e`
freezing the first phase, 189 tests collected and passing (exit 0), strict
MkDocs build clean.

Every number below is labelled with its statistic (mean/median), its
population (single seed, pooled seeds, ensemble), its data split (30-day
validation subset, 92-day validation, 98-day reused test), and its economics
version (env-v1 uncorrected, env-v2 penalized). Numbers from different
phases are never mixed silently.

---

## 1. Executive summary

The objective was a demonstrably near-optimal trading-and-dispatch controller
for a hybrid wind–PV–battery virtual power plant whose installed generation
exceeds its grid connection, trading in the German day-ahead auction, three
intraday auctions, and continuous intraday, with real reBAP imbalance
settlement — evaluated against information-equivalent optimization benchmarks
on strictly historical data. The final promoted controller is not a single
trained agent but a construction: the **mean strategic action of five frozen
SAC policies, clipped to a ±0.1 bounded residual around the analytically
rule-equivalent action, with all physical dispatch delegated to the
deterministic rule-based tracker** and every setpoint passed through an exact
feasibility projection. The strongest performance result is from the first
phase: on the 98-day test split the five-seed SAC hybrid achieved a **median**
information-equivalent gap of ≈0.07% (pooled seeds), and the best single seed
reached a test **mean** of 48,104 EUR/day versus 46,850 for the rule-based
controller — but that seed could not be identified in advance. The main
robustness result is from the second phase: the ensemble-plus-bound
construction compressed the paired test confidence interval versus rule-based
from a width of ~5,750 EUR/day (pooled seeds) to **125 EUR/day**, and the
worst single-day loss versus rule-based from five figures to **847 EUR**. RL
did **not** demonstrably beat the rule-based controller in expectation: the
promoted controller's reused-test paired mean is **−83 EUR/day (CI95 [−155,
−30])** — statistically distinguishable from zero but economically small
(≈0.2% of daily revenue) — while on winter validation it was ahead (+27
EUR/day, P(outperform) = 91%). RL matched the information-equivalent MILP on
**medians** (validation median gap −1.4%, i.e. above the MILP; reused-test
median gap +0.58%) but not on means. The main unresolved limitation is
seasonal transfer: validation is winter-only, the test window is spring, and
no protocol on this dataset can verify cross-season generalization before
deployment.

---

## 2. System and research setup

**The plant.** A hybrid park with wind and PV whose combined installed
capacity exceeds the grid connection (export limit 100 MW in the base
configuration; the oversizing ratio is a first-class config quantity), plus a
battery energy storage system (60 MWh in the base configuration, with
charge/discharge power limits, efficiencies, SoC bounds, degradation cost of
1.5 EUR/MWh throughput). Because generation can exceed the export limit,
congestion is a normal operating condition, and the framework distinguishes
**technical curtailment** (grid-forced) from **economic curtailment**
(price-driven) throughout the accounting.

**The markets.** Five sequential trading opportunities per delivery day,
modeled with exact gate closures in Europe/Berlin wall time: the day-ahead
auction (DAA, 12:00 D−1; hourly products until 2025-09-30, 15-minute products
from 2025-10-01), the three pan-European intraday auctions (IDA1 15:00 D−1,
IDA2 22:00 D−1, IDA3 10:00 D covering delivery 12:00–24:00), and continuous
intraday (IDC) modeled as a **price-taker index execution**: a trade decided
at lead time ℓ executes at the ID1 / ID3 / IDFULL volume-weighted index
depending on ℓ, plus transaction cost and volume caps (an approximation,
labelled as such — no order-book realism is claimed). Any residual between
the final contracted position and physical delivery settles at the historical
quality-assured **reBAP** (single-price German imbalance settlement).

**Control split.** The action space separates *commercial* decisions (what to
bid/trade in each market) from *physical* dispatch (battery power,
curtailment), which is the architectural hinge of the whole study: the final
controller keeps RL only on the commercial side.

**Data.** The private `iaew-marktdaten.db` (SQLite, 5.2 GB; EEX/EPEX,
ENTSO-E, Netztransparenz sources), opened strictly read-only. Site-level
renewable profiles are zone-scaled ENTSO-E actuals by default (the database
has no site-level history; Renewables.ninja profiles are supported via API
token). Forecasts are synthetic but leakage-safe: an error model fitted on the
training partition of the zone forecast-error table, applied to site actuals
as functions of `(issue_time, delivery_times)`. A byte-reproducible
**synthetic market database** serves as an automatic drop-in fallback
(resolver modes real/synthetic/auto), so CI runs fully offline; every run
records its data provenance.

**Splits (chronological, no shuffling).** The usable window where all five
markets, reBAP, and zone data overlap is **2024-06-14 → 2026-05-10** (696
days, bounded by IDA go-live and the IDC index end). Train 2024-06-14 →
2025-10-31 (~505 days, includes the DAA resolution switch); validation
2025-11-01 → 2026-01-31 (92 winter days); test 2026-02-01 → 2026-05-09 (98
late-winter/spring days). The first research phase additionally used a fixed
30-day validation subset (November) for screening — a choice that later
proved to be a weakness (§9).

---

## 3. Important data and market-model findings

The data audit (`docs/data_audit.md`) established, by verification rather
than assumption:

* **Timezones are heterogeneous per source.** ENTSO-E tables store naive
  *Europe/Berlin local* timestamps (verified by DST-day row counts: 23 rows
  on the spring day, duplicated hour in autumn — 6 duplicate timestamps in
  `day_ahead_prices`); EEX/EPEX wide tables use local product columns
  including explicit `hour 3a`/`hour 3b` DST columns; IDC and Netztransparenz
  tables are UTC. Internally everything is tz-aware UTC; Berlin wall time
  exists only in the market calendar and reporting. No day is ever assumed to
  have 96 quarter-hours (DST days have 92 or 100).
* **The EEX-derived `*_ordered` tables are corrupt and were deliberately
  avoided**: the IDA3 ordered table shifts delivery 12:00–24:00 onto
  00:00–11:45 (12-hour shift), the IDA1/2 ordered tables write 96 rows even
  on 92/100-quarter DST days, and the legacy ordered table silently drops
  both DST days each year. The project reads the wide tables and performs its
  own DST-correct reshape, pinned by regression tests against hand-checked
  days (including solar-dip timezone fingerprinting).
* **The DAA product-resolution switch** (hourly → 15-minute on 2025-10-01,
  SDAC MTU change) is detected from data, not configured, and both regimes
  are supported. It falls inside the training split by design.
* **IDA availability**: IDA1 complete in-window; IDA2 missing 13 auction
  days, IDA3 missing 10 — handled as "auction not held" events (positions
  carry forward), not interpolated.
* **IDC prices** exist only as per-product indices (ID1/ID3/IDFULL VWAPs) —
  hence the price-taker execution model; index coverage ends 2026-05-10,
  which fixes the study-window end.
* **reBAP** is the quality-assured Netztransparenz series (true German
  imbalance price, single-price regime) until 2026-06-23; observations never
  contain future reBAP (it is published months later), and it enters only
  ex-post settlement.

These findings mattered because two of them (ordered-table corruption, local
naive timestamps) would have silently produced *wrong prices at wrong hours*
— errors large enough to invalidate any RL-versus-baseline comparison — and
the leakage rules (issue-time-indexed forecasts, publication-lagged prices,
chronological splits, config-derived normalization with nothing fitted on
data) are what make the reported gaps interpretable at all. A dedicated
leakage test suite pins these properties.

## 4. Environment and simulator design

* **Event-driven simulator as the single source of economic truth.** A
  market calendar generates the exact event stream per delivery day (auction
  gate closures, IDC decision points, physical dispatch per quarter-hour,
  settlement), DST-exact. All controllers — rule-based, MILP, RL — run
  through the same engine; RL variants are forbidden (by convention and
  review) from bypassing it.
* **Append-only position book and per-component ledger.** Every trade
  appends; every euro is booked exactly once (market cash at execution,
  settlement at delivery, penalties at settlement), decomposed per market
  (DAA/IDA1/IDA2/IDA3/IDC), imbalance, transaction cost, degradation, and
  curtailment components. This decomposition later enabled the decisive
  trajectory audits (§7).
* **Commercial/physical separation.** Positions are commitments; delivery is
  physics. The battery model enforces power limits, efficiencies, and
  SoC-aware energy bounds; curtailment is bounded by available generation.
* **Feasibility projection.** Every dispatch request is projected onto the
  feasible set (grid export/import limits, SoC-aware battery bounds,
  curtailment bounds) by an exact weighted QP (KKT-multiplier bisection) or
  priority heuristics; grid power is *derived* from corrected variables,
  never clipped; every correction records requested/applied/reason; technical
  and economic curtailment are tracked separately. Eight dedicated test cases
  cover the congestion geometry.
* **Action representations are schema-versioned** (act-v1 … act-v5, §6), and
  observation size is decoupled from action dimensionality. Key invariants
  are test-pinned: the zero action in residual mode ≡ rule-based; the
  rule-equivalent parameter vector in strategic mode reproduces rule-based
  revenue to relative 1e-6.
* **Reward = real economics.** The per-step reward is the simulator's cash
  flow (scaled), with terminal battery energy valued at the day's mean DAA
  price; no shaped auxiliary rewards are part of any reported result.

Verification: 189 tests (unit/integration/leakage), including DST-day
regression tests, feasibility case geometry, accounting identities, action
schema pins, and — added in the robustness phase — fold-leakage guards,
ensemble/gate semantics, bootstrap properties, and deployment fallbacks.

---

## 5. Initial baseline results

On the fixed 30-day validation subset under the original (env-v1) economics,
mean EUR/day: **do-nothing 47,908; rule-based 50,861; information-equivalent
MILP 51,165; perfect-foresight MILP 54,994**. The perfect-foresight MILP is
an unattainable information upper bound and was never a pass/fail target.

Two findings from this stage shaped everything that followed. First, the
forecast-driven (information-equivalent) MILP is only ~0.6% ahead of the
rule-based controller — and later, on the full 92-day winter validation, it
actually fell *behind* rule-based on means (55,260 vs 55,511) while winning
medians. The reason is structural: the MILP optimizes point forecasts
aggressively, so forecast errors convert into imbalance and churn on hard
days, while the rule-based controller's conservative coverage behavior is
accidentally robust. This made "beat rule-based" the practically binding
target, with the MILP as the information benchmark.

Second, the initial PPO baseline on the raw 103-dimensional direct action
space was not merely weak but *uninformative*: evaluation oscillated between
−157k and +23k EUR/day with no trend (W&B runs `nv2qwm96`, `aesjpwo9`;
frozen action σ, rising clip fraction), and the direct-action screening
reference later confirmed ~6.9k EUR/day — far below do-nothing. This was
recorded as a formulation problem, not an algorithm problem (§6).

## 6. RL algorithm and representation study

Tested online algorithms: PPO, RecurrentPPO, SAC, TQC, TD3 (SB3/contrib),
CrossQ and SBX-SAC (JAX). Tested representations: direct incremental orders
(act-v1, 103 dims), target positions (act-v2, 103 dims), hourly targets
(act-v3, 28 dims), bounded residual around rule-based (act-v4, 28 dims), and
strategic economic variables (act-v5, 7 dims). Supporting machinery: a
deterministic Level-1 learnability environment (DAA+BESS, perfect forecasts,
MILP anchor 53,739 EUR/day), behavior cloning, replay prefill (RLPD-style),
longer budgets, risk shaping, and a formally pruned tier of offline (IQL/CQL)
and model-based (TD-MPC2, DreamerV3) methods.

The main conclusions, with mechanisms:

* **Direct incremental actions failed structurally.** The event stream is
  ~1% market decisions and ~99% dispatch steps, so market actions receive
  almost no gradient signal; and incremental semantics mean that repeating
  an intention re-trades it — the diagnosed "churn loop" showed the policy
  gaining ≈+3.8M reward share at trading events and losing ≈−5.8M at
  settlement. Credit assignment between a gate action and its settlement many
  steps later did the rest. Feasibility projection intervened in only 0.1% of
  dispatch steps — the physical layer was never the bottleneck.
* **Target-position semantics helped because they are idempotent**: stating
  the same desired position twice trades nothing. This single semantic change
  took PPO from 6.9k to ~50k EUR/day territory (best screening seed 51.4k
  mean under env-v1 — later shown to be exploit-assisted; the honest
  multi-seed replication was 50,487 / 44,725 / 48,650).
* **Strategic actions (7 economic decision variables: DAA coverage,
  arbitrage scale, IDA/IDC correction gains, tracking gain, curtailment
  threshold, SoC bias — translated deterministically through the rule-based
  structure) were the most effective formulation**: the exploration problem
  shrinks to a 7-dimensional box that contains the rule-based policy as one
  interior point, the critic sees the pre-translation action (no projection
  aliasing), and the formulation *structurally cannot* speculate on
  imbalance at scale (its env-v1 audit: imbalance share −9%).
* **Algorithm choice mattered far less than action semantics and
  architecture.** Under identical clean economics and the same strategic
  space, SAC, CrossQ, and TQC landed within ~1–2k EUR/day of each other and
  seed spreads were <1.5k; whereas changing the action semantics moved
  results by 40k+ (direct → target) and the architecture change (hybrid
  dispatch, §8) moved them by +3.5k. PPO was consistently weakest under
  env-v2 (e.g. V2 screening: PPO-hourly 38,986; PPO-target 35,333 mean on the
  30-day subset).

## 7. Reward-exploit and validity findings

The single most important validity event: under env-v1 economics, the
completed screening produced **CrossQ + hourly-target at 61,414 EUR/day mean
(30-day subset) — above the 54,994 perfect-foresight anchor**. Exceeding an
information upper bound is not a triumph; it is a red flag. Trajectory audits
(registry-recorded) showed 50–67% of profit coming from the imbalance
component (CrossQ-hourly 62%, PPO-target seeds up to 67%), with deviation
ratios up to 2.7× delivered energy: the policies had learned **imbalance
speculation** — deliberately contracting positions they did not intend to
deliver, because deviations settled at historical reBAP with no penalty and
the price-taker assumption does not push back at any volume. The accounting
was correct; the *model* was economically incomplete for judging market
operation (real balancing groups must not deviate deliberately).

Consequences, all registry-recorded: env-v1 results of speculation-capable
formulations were excluded from the operational ranking (kept and annotated
as a model finding — including the fact that the apparently-strong PPO-target
screening result did not replicate across seeds); **env-v2** added a
25 EUR/MWh deviation penalty — a documented proxy for balancing-group
compliance obligations — to settlement for *all* controllers, and all anchors
were recomputed (30-day subset: do-nothing 46,734 / rule-based 50,042 /
info-MILP 50,679 / perfect 54,704).

Two further validity-relevant mechanisms were then isolated:

* **Corner-optimum unreachability**: rule-equivalent behavior mapped to
  correction gains = 1.0 → raw action +1.0, the corner of the squashed-
  Gaussian action box, which entropy-regularized policies cannot reach. This
  explained why prefill, BC, budget, and entropy interventions all converged
  to the same sub-baseline attractor. Fix: `strategic_gain_max = 1.25`
  interiorizes the optimum (medians +0.7k, modest but real).
* **Under-priced reBAP tail risk**: even penalized, risk-neutral policies
  carried 3–6× rule-based's deviation (150–195 vs 26–73 MWh/day) and volatile
  reBAP days produced five-figure paired losses (worst 5 of 30 validation
  days = 46% of the remaining gap). The architectural fix — **deterministic
  rule-based dispatch under RL market control** — eliminated the channel
  rather than repricing it, and produced the breakthrough (§8).

## 8. First final RL result (`EMPIRICAL_FRONTIER_REACHED`, tag `rl-frontier-v1`)

The winning first-phase formulation (V6): SAC on strategic actions (act-v5,
gain_max 1.25) + deterministic rule-based dispatch, env-v2 economics, 300k
steps/seed, five seeds, checkpoint selected by evaluation callback on the
30-day validation subset.

**Validation (30-day subset, pooled 5 seeds)**: mean 49,449 / median 51,315
EUR/day; per-seed validation means 48,531–50,599; median info-gap 0.71%;
paired bootstrap vs rule-based (50,042 mean) CI95 **[−1,157, −6]** — narrowly
but entirely below zero.

**Test (98 days, one locked evaluation, env-v2):**

| Controller | Mean EUR/day | Median EUR/day |
|---|---|---|
| RL V6, pooled 5 seeds | 45,147 | 48,174 |
| — per-seed means | 40,736 / 46,975 / 48,104 / 43,783 / 46,139 | (medians 47,590–48,660) |
| do-nothing | 42,512 | 41,462 |
| rule-based | 46,850 | 48,464 |
| info-equivalent MILP | 47,599 | 49,294 |
| perfect-foresight MILP (upper bound) | 52,007 | 52,573 |

Paired RL−rule-based (pooled seeds): mean −1,703 EUR/day, CI95
**[−5,085, +667]** — statistically indistinguishable from rule-based; median
info-gap **+0.07%** (a *median* statistic — it does not transfer to means,
where the pooled gap vs the info-MILP was ≈ −2,450 EUR/day ≈ 5.2%).

`SUCCESS_NEAR_OPTIMAL` was **not** declared, for two reasons stated in the
report: the lower confidence bound vs the best non-RL baseline was −5,085
(criterion requires ≥ 0 or at least clean indistinguishability plus
consistency), and the five-seed **consistency criterion failed in the most
instructive way possible** — the validation-best seed (50,599 validation
mean) was the *worst* test generalizer (40,736), while seed 2 quietly beat
rule-based on test means (48,104). Thirty November validation days
over-select; seed-level generalization variance, not any identified
formulation defect, dominated the residual gap. That diagnosis defined the
second phase.

## 9. Robustness phase (branch `research/robust-rl-selection`)

Design discipline first: the phase plan (`docs/robust_rl_research_plan.md`)
fixed the selection rules, score coefficients, and evaluation protocol
*before* any fold was evaluated; the 98-day test split — already examined —
was quarantined for one pre-registered read-out; and Phase 0 verified
reproducibility (all five frozen checkpoints reproduce their recorded test
revenues to the cent).

* **Blocked temporal validation (RQ1: supported).** All 35 checkpoints (5
  seeds × 7 saved checkpoints) were evaluated on all 92 validation days,
  split into 6 contiguous blocks. Variance decomposition of validation means:
  seed effect std 517, checkpoint effect std 314, seed×checkpoint interaction
  403 EUR/day — *both* selection layers of the old procedure contributed
  error, and the best checkpoint differs per seed. **Early checkpoints
  generalize better**: mean validation revenue peaks at 50k steps (55,130)
  and drifts down to 300k (54,254); three of five seeds have their optimum at
  50k. Leave-one-block-out rule comparison: the pre-registered risk-adjusted
  score (mean − 0.5·fold-std − 0.5·tail-loss − 0.25·shortfall) is the only
  rule family with *positive* mean held-out regret vs rule-based (+354
  EUR/day; oracle gap 1,394), while highest-mean (−704), highest-median
  (−752), and highest-worst-fold (−2,268) select worse. Read-only test
  consistency check (computed after rule locking): Spearman between the
  risk-adjusted score and the recorded test means of the five best
  checkpoints is **+0.80** (raw validation mean: +0.40), and the score ranks
  the pathological seed 0 last where the old 30-day window ranked it first.
  Action statistics predict regret: intraday over-trading correlates with
  losing (IDA gain ρ=+0.71, IDC gain +0.69), day-ahead coverage with winning
  (ρ=−0.71).
* **Ensembles (RQ2: supported).** On the 92-day validation, every
  action-space ensemble of the five eval-best checkpoints beat every
  individual member on mean revenue. The plain **mean ensemble** is best:
  55,635 mean / 56,916 median (members 54,305–55,110 mean), paired regret vs
  rule-based **+124 EUR/day**, P(day beats rule-based) 0.52, CVaR₁₀% of daily
  regret **−1,822** vs −6,374…−14,788 for members, downside exposure 309 vs
  1,151–2,380 EUR/day. Trimmed-mean and LOBO-weighted variants were not
  better than the plain mean; the median-action ensemble was worst.
* **Disagreement gate and the control experiment (H3: refuted).** Ensemble
  disagreement does *not* predict bad days: Spearman(u, daily regret) =
  +0.11 (wrong sign for a safety signal); high-disagreement days had
  *higher* mean regret. The hard disagreement gate (fallback at the 80th
  percentile) still improved tails (CVaR −1,032, P(outperform)=93%), but a
  **random-fallback control** at the same 20% rate recovered most of that
  improvement (CVaR −1,441): the value is bounded dampening toward the rule
  action, not the signal.
* **Bounded residual (RQ3: supported).** Clipping the ensemble action to
  ±0.1 per dimension around the rule-equivalent anchor gave, on validation:
  mean 55,538 / median 56,988, regret +27, CVaR₁₀% −437, downside exposure
  **77 EUR/day**, max daily loss 1,430, P(day beats RB) 0.57, block-bootstrap
  P(outperform) 91%. The pre-registered risk-adjusted rule, applied to all
  composites, selected this variant (score 50,966) as the locked candidate.
* **Failure days and regimes (RQ4 answered; RQ5 pruned).** Failure days
  over-represent negative DAA prices (lift 4.6–5.8, but base rate 1% — one
  winter day), low renewables (1.8–2.3), and price volatility (1.3–2.0);
  reBAP volatility is largely neutralized by the hybrid architecture, and
  forecast error is *not* enriched. A negative-price regime gate triggered on
  1 of 92 validation days — unvalidatable — so regime conditioning was
  formally pruned rather than fitted. A day-level Mahalanobis OOD score was
  a weak predictor (Spearman −0.05) and was recorded, not adopted.
* **Retraining pruned.** With ENSEMBLE_SUCCESS conditions already met by an
  existing-policy construction, bounded-residual/regime/regret-aware SAC
  retraining was formally pruned (the gate already caps the targeted tail;
  BC-style initialization had washed out twice; regime evidence was n=1).
* **Stronger MILP benchmarks (Phase 6).** Turnover-penalized MILPs improve
  the classic info-MILP (mean 55,304 vs 55,260; CVaR regret −4,508 vs
  −5,621); forecast-derated variants lose mean. None approaches the RL
  composites on tails (−437 for the gate), while the MILPs hold the highest
  *medians* (~58.2k) — they win normal days and lose hard days.
* **Zero-shot sensitivity (RQ6: supported, with one boundary).** With no
  retraining, the locked candidate stays ahead of rule-based under grid
  export ±20% (+23/+24 EUR/day), BESS energy ×0.5/×2 (+36/+27), and degraded
  persistence forecasts (+33); it dips only under perfect forecasts (−17,
  where corrections have nothing to add). Its deviation volume (37.8 MWh/day)
  matches rule-based (37.6), so the exact analytic deviation-penalty sweep
  (0/10/25/50/100 EUR/MWh) preserves the ordering — except at 100 EUR/MWh,
  where the low-deviation info-MILP (21.8 MWh/day) overtakes all
  trading-heavier controllers.

## 10. Final promoted deployment controller

Let $a_t^{(i)} \in [-1,1]^7$ be the deterministic strategic action of frozen
SAC policy $i \in \{1,\dots,5\}$ (the five eval-best checkpoints of the V6
seeds), and let $a^{\mathrm{rule}}$ be the analytically derived
rule-equivalent action ($[2/1.2-1,\; 1,\; 0.6,\; 0.6,\; \cdot,\cdot,\cdot]$
under gain_max = 1.25; a pinned test verifies it reproduces rule-based
revenue). The deployed action at every market event is

$$
a_t^{\mathrm{deploy}} \;=\; a^{\mathrm{rule}} \;+\;
\operatorname{clip}\!\Big(\tfrac{1}{5}\textstyle\sum_{i=1}^{5} a_t^{(i)}
\;-\; a^{\mathrm{rule}},\; -\Delta,\; +\Delta\Big),
\qquad \Delta = 0.1 \text{ per dimension.}
$$

Physical dispatch is **not** RL-controlled: the deterministic rule-based
tracker converts positions into battery and curtailment setpoints, and the
exact feasibility projection enforces grid, SoC, and curtailment constraints
on every quarter-hour. An operational wrapper (`envs/deployment.py`) adds a
schema-version check at startup, a missing-data fallback (non-finite
observations → rule action), an out-of-range OOD fallback, and a complete
replayable decision log (observation digest, member proposal, gated action,
reason, model identities).

Why this beats deploying one selected seed: (i) it removes seed selection
entirely — the previous phase's dominant failure mode, where the
validation-chosen seed landed 6,114 EUR/day below rule-based on test; (ii)
averaging cancels idiosyncratic per-seed mistakes exactly on the days that
produced tail losses; (iii) the residual bound converts "RL might do
something unbounded" into "rule-based ± a small, auditable correction",
making worst-case behavior a design parameter rather than an empirical hope;
(iv) every component is deterministic and frozen, so the controller is
exactly reproducible.

## 11. Final robustness results (reused test, pre-registered one-shot)

One deterministic pass over the 98 reused test days (2026-02-01 →
2026-05-09), env-v2 economics, compared against the recorded baselines:

| Quantity (reused test) | Value |
|---|---|
| Promoted controller mean / median | **46,767 / 48,517 EUR/day** |
| Rule-based mean / median | 46,850 / 48,464 |
| Info-MILP mean / median | 47,599 / 49,294 |
| Perfect-foresight mean / median (upper bound) | 52,007 / 52,573 |
| Paired mean vs rule-based (block bootstrap) | **−83 EUR/day, CI95 [−155, −30]** |
| P(outperform rule-based in the mean) | 0.2% (day-level win rate 31%) |
| Median info-gap | **+0.58%** (median statistic) |
| CVaR₁₀% of daily regret | −569 EUR |
| Downside exposure | 141 EUR/day |
| **Maximum single-day loss vs rule-based** | **847 EUR** |
| Paired CI width | **125 EUR/day** (previous pooled seeds: 5,752) |

The terminal state is **`ENSEMBLE_SUCCESS`**, not `ROBUST_SUCCESS`, and the
distinction matters. ENSEMBLE_SUCCESS requires that an ensemble/gate built
from existing policies materially improves robustness, removes dependence on
one favorable seed, and is confirmed across temporal folds — all of which
hold: the paired CI narrowed 46-fold, the worst day improved from five-figure
losses to 847 EUR, the median beats rule-based (48,517 vs 48,464), and no
oracle choice exists anywhere in the loop. ROBUST_SUCCESS would additionally
require the final method to be at least statistically competitive with
rule-based in expectation on the final read-out — and here the reused-test
paired mean is **statistically distinguishable from zero on the negative
side** ([−155, −30]), i.e. the promoted controller is a statistically
significant but economically small ~0.2% *below* rule-based in mean revenue
on the spring test window, even though it was ahead on winter validation
(+27, P(outperform)=91%). The ensemble improved robustness decisively; it did
**not** demonstrate higher expected revenue than the rule-based controller.

## 12. Mechanism-level findings

| Finding | Evidence | Implication |
|---|---|---|
| Action semantics dominate dimensionality and algorithm choice | Direct-PPO 6.9k → target-semantics ~50k on the same 103 dims; SAC/CrossQ/TQC within ~1–2k of each other; strategic 7-dim best | Design the action space around economic decisions, not around raw tensors |
| Direct incremental actions cause churn | Audit: ≈+3.8M reward share at trading events, ≈−5.8M at settlement; 768:8 dispatch:gate event imbalance | Idempotent (target/strategic) semantics are a prerequisite, not an optimization |
| Rule-like behavior can sit at the action-box corner, unreachable for squashed Gaussians | All of prefill/BC/budget/entropy converged to the same sub-baseline attractor; gain interiorization (+0.7k medians) fixed it | Check where the reference policy lives in action space before blaming the algorithm |
| Unpenalized reBAP settlement invites imbalance speculation that invalidates results | CrossQ 61.4k > perfect-foresight 55.0k; 50–67% of profit from imbalance; deviation up to 2.7× delivery | Any RL market study must audit the profit decomposition; beating an information bound is a bug signal |
| Deterministic dispatch under RL markets is the decisive architecture | V6: +3.5k mean over pure-RL variants; deviation 94 vs 150–195 MWh/day; validation seed 0 beat rule-based on both mean and median | Keep RL on commercial decisions; let deterministic control deliver |
| Ensembling in action space removes seed variance | Every ensemble > every member (92-day val); CVaR regret −1,822 vs −6.4k…−14.8k; test CI width 125 vs 5,752 | Deploy all seeds, select none |
| The residual bound is insurance, priced at ~0.2% of revenue | Test: max daily loss 847 EUR, downside 141/day, paired mean −83 [−155,−30] | Worst-case behavior becomes a design parameter; the premium is measurable |
| Disagreement is not a causal safety signal here | Spearman(u, regret)=+0.11; random 20% fallback recovers most of the gate's tail benefit | Don't dress dampening up as uncertainty quantification |
| Seasonal distribution shift is the unresolved risk | Validation edge +27 (P 91%, Nov–Jan) inverted to −83 (Feb–May); winter-only validation cannot test this | Prospective/seasonal validation is the binding next requirement |
| Negative-price periods are the best-attributed improvement opportunity | Failure-day lift 4.6–5.8; only 1/92 winter validation days → unfittable | Needs spring/summer validation data, not more winter modeling |

## 13. What failed and why

**Experimentally shown to fail (multi-seed, registry-recorded):**

* *Direct-action PPO (act-v1)*: churn loop + event imbalance + long-horizon
  credit assignment; ~6.9k EUR/day vs 47.9k do-nothing (env-v1, 30-day
  subset).
* *Target-position formulations under env-v1 economics*: learned imbalance
  speculation (50–67% profit share); results excluded from operational
  ranking; the flagship screening number did not replicate across seeds
  (50,487 / 44,725 / 48,650).
* *PPO generally under env-v2*: 35–44k across variants (30-day subset means)
  vs SAC/CrossQ 44–47k.
* *Behavior-cloned initialization* (twice: V3 pure, V7 hybrid): BC start at
  rule-based washes out during entropy-regularized fine-tuning — V7 means
  47,964–49,465 vs V6's 48,531–50,599 on the same subset.
* *Median-action ensemble*: worst ensemble variant (−137 mean regret vs +124
  for the mean ensemble, 92-day validation).

**Statistically inconclusive (differences within seed noise):**

* *Replay prefill* (RLPD-style): +0.2–0.7k on 30-day means, within noise,
  for both SAC and CrossQ.
* *Risk shaping* (train at 50 EUR/MWh penalty, evaluate at 25): 45,237–45,834
  — no gain over cold-entropy scratch.
* *Disagreement gating beyond random dampening*: Gate A beat the random
  control on every tail metric, but by margins (CVaR −1,032 vs −1,441) far
  smaller than the gate-vs-nothing effect.
* *LOBO-weighted ensembles*: 55,579 vs 55,635 for the plain mean.

**Deprioritized / formally pruned with stated reasons (not disproven):**

* *1M-step budget run*: failed twice on code-skew exceptions during live
  development, then pruned as mooted by the V6 architecture result (the
  plateau it was probing no longer existed).
* *Offline IQL/CQL*: the BC arm already started at the dataset's best policy
  and lost it during fine-tuning; support-mismatch risk documented.
* *TD-MPC2 / DreamerV3*: sample efficiency and credit assignment were shown
  not to be binding (300k-step runs converge with sub-1.5k seed spreads).
* *Regime conditioning and regime-aware retraining*: the dominant regime
  (negative prices) occurs on 1 of 92 validation days — unfittable and
  unvalidatable in-window.
* *Critic-weighted ensembles*: SAC Q-values are entropy- and scale-shifted
  across seeds; calibration was not established, so the variant was skipped
  per the plan's precondition.

**Not tested fully:** retrained bounded-residual/regret-aware SAC (pruned on
evidence economy, not falsified); TD3 beyond smoke level; RecurrentPPO at
scale; adaptive residual bounds.

## 14. Scientific claims

**Supported claims**

1. On this problem, action-space semantics and control architecture dominate
   RL algorithm choice by an order of magnitude (§12, rows 1–2, 5).
2. Unpenalized historical-reBAP settlement makes imbalance speculation the
   rational optimum for expressive action spaces; profit-decomposition audits
   are necessary for validity (env-v1 evidence; reproducible).
3. A hybrid controller — RL strategic market decisions, deterministic
   dispatch — reaches a **median** information-equivalent gap of ≈0.07%
   (pooled seeds, 98-day reused test) and beats rule-based on winter
   validation medians and means for its best seeds.
4. Blocked temporal validation with a risk-adjusted selection score selects
   materially better than short-window mean selection (LOBO +354 vs −704
   EUR/day held-out regret; score-test Spearman +0.80 vs +0.40).
5. Action-space ensembling of independently trained seeds removes
   seed-selection risk and cuts tail regret several-fold without retraining
   (every ensemble > every member; test CI width 125 vs 5,752 EUR/day).
6. A ±0.1 bounded residual around the rule-equivalent action caps the
   worst observed single-day loss vs rule-based at 847 EUR on 98 test days,
   at a measured mean cost of ~0.2% of revenue.
7. The ensemble+bound construction is zero-shot robust to ±20% grid limits,
   ×0.5/×2 storage sizing, and degraded forecasts on winter validation.

**Unsupported claims (do not make these)**

* "RL consistently beats rule-based control." False on test means (pooled
  −1,703 [−5,085, +667]; promoted controller −83 [−155, −30]); true only for
  medians and for particular seeds/periods.
* "RL consistently beats the information-equivalent MILP." Median-level
  only, and period-dependent; on test means the MILP is ahead of everything
  RL produced (47,599).
* "The best seed can be selected reliably from a short validation window."
  Directly falsified: the 30-day-window choice was the worst test seed.
* "Policy disagreement is a calibrated uncertainty measure." Refuted
  (ρ=+0.11 wrong-signed; random-fallback control).
* "The controller is proven robust across all seasons." Winter-only
  validation cannot support this; the validation edge inverted on the spring
  test window.
* "The 0.07% / 0.58% gaps apply to mean revenue." They are **median**
  statistics; mean gaps vs the info-MILP are ≈5.2% (pooled seeds) and ≈1.7%
  (promoted controller) on the reused test.

**Conditional claims**

* The revenue ordering RL-composite ≥ rule-based ≥ info-MILP holds at the
  25 EUR/MWh deviation penalty and (analytically, zero-shot) down to 0; at
  100 EUR/MWh the low-deviation info-MILP wins — high-compliance-cost
  regimes need retraining or MILP control.
* The +27 EUR/day validation edge is a Nov–Jan statement; the −83 EUR/day
  test result is a Feb–May statement; neither transfers to summer, where
  negative-price days (the highest-lift failure mechanism) concentrate.
* All results assume price-taker IDC execution at published indices with
  volume caps, synthetic-but-leakage-safe site forecasts, and zone-scaled
  site profiles; order-book impact and site-specific forecast quality are
  outside the model.
* Sizing robustness was shown for grid ±20% and BESS ×0.5/×2 around the base
  configuration, zero-shot; larger excursions were not tested.

## 15. Practical implications

For an operational EMS, the study's architecture translates directly:

* **RL belongs on strategic decisions, not raw physical actions.** The
  7-dimensional strategic layer is where learning added value; every attempt
  to learn dispatch-adjacent behavior added deviation risk that the market
  eventually priced.
* **Deterministic dispatch should remain deterministic.** Position tracking
  by the rule-based controller plus exact feasibility projection was the
  single largest performance and safety improvement in the study, and it
  keeps compliance behavior (deviation ≈ rule-based level) certifiable.
* **Fallbacks are structural, not optional.** The deployment wrapper's
  missing-data and out-of-range fallbacks to the rule-equivalent action, the
  schema check, and the replayable decision log mean the system degrades to
  the incumbent controller, never to undefined behavior.
* **Ensembles over oracle seeds.** Train several seeds, freeze them, average
  their actions. Retrospectively picking the best seed is a lottery this
  study lost once, publicly (40,736 vs 48,104 EUR/day on identical test
  days).
* **The residual bound is an insurance dial.** At Δ=0.1 the measured premium
  was ~83 EUR/day against a hard 847 EUR worst-day cap (test); widening
  toward the ungated ensemble buys expectation (+124 validation regret) at
  the cost of ±5k daily swings. This is a risk-appetite decision, not a
  research question.
* **The rule-based controller remains the reference** — it is the fallback
  target, the anchor of the residual bound, the regret benchmark, and on
  current evidence the mean-revenue frontier on unseen seasons. Any deployed
  RL layer should be evaluated as a bounded modification of it.

## 16. Next research steps (prioritized)

1. **Prospective evaluation on genuinely new post-May-2026 data** (new IDC
   index coverage permitting): the only way to make an untouched-test claim
   again — pre-register the promoted controller, then evaluate once.
2. **Summer and negative-price regimes**: extend the market window to
   include high-PV months; the failure-day analysis marks negative-price
   handling (lift 4.6–5.8) as the best-attributed improvement opportunity.
3. **Seasonal-transfer validation**: rolling-origin evaluation across season
   boundaries (train→winter-val→spring-val→summer-val) so selection sees the
   shift the current protocol structurally cannot.
4. **Pre-registered sequential evaluation**: adopt the phase-2 discipline
   permanently — locked rules, one-shot read-outs, registry-recorded
   decisions — as the default lab protocol.
5. **Negative-price-specific strategy**: a targeted curtailment/charging
   policy for forecast-negative-price days, validated on data where such
   days exist in double digits.
6. **Stronger stochastic/risk-aware information-equivalent MILP**:
   scenario-based or CVaR-constrained variants (the turnover-penalized MILP
   already closed part of its tail gap); the benchmark should be as robust
   as the controller it judges.
7. **Adaptive residual bounds without overfitting**: test schedule- or
   volatility-indexed Δ against the fixed-Δ baseline under the blocked
   protocol, with the random-control methodology to separate signal from
   dampening.
8. **Publication preparation**: the mechanism ladder (speculation →
   corner-optimum → tail risk → seed variance), the selection/ensemble
   protocol, and the honest negative results are the paper; the repository
   already contains the reproducible artifacts.

A further broad algorithm sweep is *not* recommended: algorithm choice was
repeatedly shown to be a second-order factor on this problem.

## 17. Publication-ready takeaway

**Conclusion paragraph.** This study set out to learn near-optimal market
operation for an oversized hybrid wind–PV–battery plant under realistic
German market microstructure and found that the path to a deployable result
ran not through algorithms but through validity and architecture: an
economically incomplete settlement model was exploited exactly as theory
predicts; an action-space corner made the incumbent policy unreachable; a
risk-neutral objective bought revenue with tail risk the market eventually
priced; and a short validation window selected precisely the wrong model.
Each mechanism was identified by controlled intervention, fixed, and
verified. The final controller — five frozen SAC policies averaged in a
7-dimensional strategic action space, bounded to a small residual around the
rule-equivalent action, above deterministic dispatch — matches the
information-equivalent MILP on median days, tracks the incumbent rule-based
controller within ±0.2% of mean revenue with a hard cap on daily losses, and
is exactly reproducible. What remains open is not capability but transfer:
demonstrating that any learned edge survives a season it has never seen.

**Abstract-style result paragraph.** On 98 held-out spring days (reused test
split, penalized-imbalance economics), the promoted ensemble controller
earned a mean 46,767 EUR/day versus 46,850 for the rule-based incumbent
(paired difference −83 EUR/day, CI95 [−155, −30]) and 48,517 versus 48,464 on
medians, with a median information-equivalent gap of +0.58%, a worst
single-day shortfall of 847 EUR, and a paired confidence interval 46× narrower
than the previous five-seed procedure; on 92 winter validation days the same
construction outperformed the incumbent with probability 0.91 (block
bootstrap, +27 EUR/day) and its ungated variant by +124 EUR/day. The pooled
five-seed hybrid SAC formulation it is built from attains a median
information-equivalent gap of ≈0.07% on the same test days (means: 45,147
pooled, 40,736–48,104 per seed), quantifying a seed-selection risk that the
ensemble eliminates by construction.

**Five key contributions.**

1. A validity methodology for market-RL: profit-decomposition audits that
   caught reward exploitation (policies exceeding a perfect-foresight bound
   via imbalance speculation) and a penalized-settlement correction applied
   uniformly to all controllers.
2. A mechanism ladder — imbalance speculation, corner-optimum
   unreachability, under-priced tail risk, seed-selection variance — each
   diagnosed with controlled evidence and fixed by intervention rather than
   hyperparameter search.
3. The hybrid architecture result: RL strategic market control above
   deterministic dispatch closes the information-equivalent gap on medians
   while keeping deviation behavior at compliance level.
4. A selection-and-ensembling protocol (blocked temporal validation,
   pre-registered risk-adjusted scoring, mean-action ensembles, bounded
   residual) that converts an unreliable per-seed result into a reproducible
   deployable controller, validated by a pre-registered one-shot read-out.
5. A fully reproducible research object: event-exact simulator, leakage test
   suite, synthetic-data CI fallback, registry-recorded experiment
   provenance including all negative results, and frozen tagged phases.

**Five limitations.**

1. Validation is winter-only and the test window is spring; seasonal
   transfer is unverified, and the validation edge did not survive it.
2. The test split was reused across phases; the final read-out is a
   pre-registered confirmation, not an untouched-test claim.
3. IDC execution is a price-taker index model with volume caps — no
   order-book impact; site profiles are zone-scaled and forecasts synthetic
   (leakage-safe, but not site-measured).
4. Results are calibrated to the 25 EUR/MWh deviation-penalty regime; at
   100 EUR/MWh the low-deviation MILP dominates zero-shot.
5. Mean-level RL superiority over the rule-based incumbent was not
   demonstrated on unseen data — median-level parity and tail-risk
   dominance were.

**Plain-language takeaway.** After 49 registered experiments, the honest
result is: reinforcement learning did not out-earn a well-built rule-based
operator on average — but it matched it on typical days, occasionally beat it
by a lot, and, once wrapped in an ensemble with a hard safety leash, it
became a controller you could actually deploy: never more than ~850 EUR/day
worse than the incumbent on the worst observed day, usually within a few tens
of euros, and with every decision auditable and reproducible. The expensive
lessons — that the market model can be gamed, that the obvious way to pick
"the best model" picks the worst one, and that averaging five mediocre-looking
policies beats trusting one good-looking one — are worth more than the
revenue difference.
