# Robust RL research log

Chronological record of the robustness/selection research phase
(branch `research/robust-rl-selection`, plan in
[robust_rl_research_plan.md](robust_rl_research_plan.md)). The previous
phase is frozen at tag `rl-frontier-v1`.

## 2026-07-18 — Phase 0: reproducibility gate passed

* Full test suite: 158 passed, 0 failed (real market DB present).
* All six frontier artifacts verified in place; nothing overwritten.
* The five final V6 checkpoints (`best_model.zip`) reload and reproduce
  their recorded per-day test revenues to the cent on a 4-day-per-seed
  spot check (20/20 exact matches).
* Tagged `rl-frontier-v1` at `17b002e`; created branch
  `research/robust-rl-selection`.
* Checkpoint inventory: every seed kept 50k/100k/150k/200k/250k/300k
  step checkpoints plus the eval-best model → 35 candidates for the
  selection study.
* Research plan written (`docs/robust_rl_research_plan.md`); selection
  score coefficients and rules fixed before any fold evaluation.

## 2026-07-18 — Phase 1 started: blocked temporal validation

* `evaluation/blocked_validation.py`: contiguous ordered blocks,
  fold/CVaR/regret metrics, risk-adjusted selection score, six
  predefined selection rules, leave-one-block-out reliability protocol.
  9 unit tests including a fold-leakage guard (a candidate that shines
  only inside the held-out block must not be selectable).
* `evaluation/checkpoint_matrix.py`: cached deterministic rollouts of
  all 35 checkpoints + 3 baselines over the 92 validation days
  (6 spawn workers), recording per-day episode metrics and raw
  strategic actions per market event (1 DAA, 3 IDA, 32 IDC,
  96 dispatch steps/day) for the later disagreement analysis.
* Full matrix evaluation launched (~3,400 episodes).

## 2026-07-18 — Phase 1 complete: selection is fixable (RQ1: yes)

35 candidates x 92 validation days, 6 contiguous blocks. Baselines on the
same days: rule-based 55,511; info-MILP 55,260; do-nothing 52,726 EUR/day.

* **Seed vs checkpoint variance**: seed effect std 517, checkpoint effect
  std 314, interaction 403 EUR/day — checkpoint choice matters almost as
  much as seed choice, and the best checkpoint differs per seed. Both
  layers of the previous procedure (eval-best checkpoint, 30-day window)
  contributed error.
* **Early checkpoints generalize better**: mean validation revenue is
  highest at 50k steps (55,130) and declines monotonically-ish to 300k
  (54,254). Three of five seeds have their per-seed optimum at 50k.
* **Selection-rule reliability (LOBO)**: the risk-adjusted score (and its
  close relative lowest-downside) is the only rule family with *positive*
  mean held-out regret vs rule-based (+354 EUR/day; gap-to-oracle 1,394;
  best block within 8 EUR of oracle). Highest-mean (−704) and
  highest-worst-fold (−2,268) select worse. **Locked rule:
  risk_adjusted** (coefficients fixed in the plan before evaluation).
* **Action statistics predict regret**: Spearman vs mean regret — DAA
  coverage −0.71, DAA arbitrage scale −0.60, IDA gain +0.71, IDC gain
  +0.69. Intraday over-trading (gain > 1) is the signature of poorly
  generalizing checkpoints; full day-ahead coverage is protective.
* **Read-only test read-out** (previous phase's recorded results, five
  eval-best checkpoints, computed after rule locking): Spearman(val mean,
  test mean) +0.40, Spearman(risk-adjusted score, test mean) **+0.80**.
  The score ranks the pathological seed 0 last; the old 30-day window
  ranked it first.
* Full-validation selections: 4 of 6 rules pick `ckpt_seed2_050k`; its
  test performance stays unknown until the Phase-8 locked confirmation.
* Report: `reports/checkpoint_selection_analysis.md`; artifacts under
  `artifacts/robust_selection/`.

## 2026-07-18 — Phase 2 complete: ensembles work, disagreement doesn't (RQ2: yes, H3: refuted)

Four action-space ensembles of the five eval-best checkpoints, 92
validation days:

* **Every ensemble beats every individual member on mean revenue.** The
  plain mean ensemble is best: 55,635 EUR/day (members 54,305–55,110),
  median 56,916, mean regret vs rule-based **+124 EUR/day** (members:
  −401 to −1,206), P(day beats rule-based) 0.52.
* **Tail-risk collapse**: CVaR₁₀% of daily regret improves from
  −6,374…−14,788 (members) to **−1,822** (mean ensemble); downside
  exposure from 1,151–2,380 to **309 EUR/day**. Averaging in action
  space cancels idiosyncratic per-seed mistakes on exactly the days
  that used to produce five-figure losses.
* Ranking: mean > trimmed-mean ≈ validation-weighted > median. The
  LOBO-weighted variant does not beat the plain mean — member quality
  differences are too small and weighting adds selection noise.
* **H3 refuted**: disagreement does not predict poor ensemble days —
  Spearman(u, daily regret) = **+0.11** (wrong sign for a safety
  signal); high-disagreement days have *higher* mean regret (+262 vs
  −14) and *lower* P(negative regret) (39% vs 57%). Per the plan's
  precondition, disagreement-based Gates A/B lose their justification;
  they are still evaluated for completeness, alongside the
  disagreement-independent bounded-residual Gate C.
* Report: `reports/ensemble_analysis.md`; `ensemble_results.json`,
  `disagreement_analysis.json`, `ensemble_weights.json`.

## 2026-07-18 — Phase 3 complete: bounded residual wins (RQ3: yes)

Seven gate variants around the mean-ensemble proposal, LOBO-calibrated
thresholds, plus two controls. Block bootstrap vs rule-based (92 days):

* ensemble (ungated): mean +124, CI [−178, +457], **P(outperform) 81%**
* gate_a_q80 (disagreement fallback): +105, CI [−47, +283], **93%**
* gate_c_r0.1 (bounded residual): +27, CI [−12, +71], **91%**,
  downside exposure 77 EUR/day (4x better than ungated), max daily loss
  vs rule-based 1,430 (vs 4,844), P(day beats rule-based) 0.57.
* All composites beat the info-MILP on the median day (info gap
  −1.1% … −1.6%).
* **Random-fallback control**: rule fallback on a random 20% of events
  recovers most of Gate A's tail benefit (CVaR −1,441 vs −1,032) —
  the safety value is bounded dampening toward the rule action, not the
  disagreement signal itself (consistent with H3's refutation).
* **Locked candidate** (pre-registered risk-adjusted rule over all
  composites, selected before any test contact): **`gate_c_r0.1`** —
  mean ensemble of the five eval-best checkpoints, per-dimension
  residual bound 0.1 around the rule-equivalent action, deterministic
  rule-based dispatch. Runner-up: gate_a_q80.
* Artifact: `safety_gate_results.json`.

## 2026-07-18 — Phase 4 complete: mechanisms identified, regime features pruned (RQ4 answered, RQ5 pruned)

* Failure days over-represent negative DAA prices (lift 4.6–5.8, but 1%
  base rate), low renewables (1.8–2.3), price volatility (1.3–2.0).
  reBAP volatility is largely resolved by the hybrid architecture;
  forecast error is not enriched — the residual gap is market-side.
* Gate E (negative-price regime fallback) triggers on **1 of 92**
  validation days: the strongest mechanism is unvalidatable in a winter
  validation window. Regime conditioning formally pruned; documented as
  the first candidate for a future phase with spring validation data.
* Reports: `failure_day_analysis.md`, `regime_analysis.md`;
  `artifacts/failure_days/`.

## 2026-07-18 — Phase 6 complete: stronger MILPs confirm the promotion

Turnover-penalized MILP (0.5/2 EUR/MWh churn penalty) improves the
classic info-MILP (mean 55,304 vs 55,260; CVaR regret −4,508 vs −5,621);
forecast derating (0.95/0.9) costs mean revenue. No MILP variant
approaches the RL composites on mean (best MILP 55,304 vs gate 55,538 /
ensemble 55,635) or tail (CVaR −4,438 at best vs −437 for the gate).
Notably the MILPs hold much *higher medians* (~58.2k): they win normal
days and lose hard on bad days. The RL composite's advantage is
robustness, not median-day optimization. Report:
`reports/robust_milp_analysis.md`.

## 2026-07-18 — OOD signal (§19): weak, not promoted

Mahalanobis distance of validation days from the 505-day training
feature distribution (price level/volatility, reBAP volatility,
renewable level, forecast error): Spearman vs the promoted controller's
daily regret **−0.05**; top-quartile-OOD days average −89 EUR/day regret
vs +66 for the rest (mean-level only, not rank-consistent). Recorded as
a candidate fallback trigger, not adopted. Artifact:
`artifacts/robust_selection/ood_analysis.json`.

## 2026-07-18 — Phase 7 complete: zero-shot robustness confirmed (RQ6: yes, with one boundary)

Locked candidate vs rule-based, zero-shot (trained on base config), 92
validation days per scenario:

* Grid export limit ×0.8 / ×1.2: **+23 / +24 EUR/day**
* BESS energy ×0.5 / ×2: **+36 / +27 EUR/day**
* Forecast quality — persistence (degraded): **+33**; perfect: −17
  (with perfect foresight the rule-based plan is already optimal-ish and
  RL corrections have nothing to add).
* Deviation-penalty sweep 0→100 EUR/MWh (analytic, exact): the candidate
  stays ahead of rule-based at every penalty (identical deviation
  profiles, 37.8 vs 37.6 MWh/day — deterministic dispatch pins the
  deviation behavior). **Boundary**: at 100 EUR/MWh the info-MILP's
  low-deviation profile (21.8 MWh/day) overtakes all trading-heavy
  controllers (53,622 vs 52,703). The promotion claim is calibrated to
  the 25 EUR/MWh compliance regime; high-penalty regimes would require
  retraining (pruned this phase).
* Artifact: `artifacts/robust_selection/sensitivity_results.json`.

## 2026-07-18 — Phase 8 protocol (pre-registered before any test contact)

* Candidate (locked in Phase 3): `gate_c_r0.1` — mean-action ensemble of
  the five frozen eval-best checkpoints, per-dimension bounded residual
  0.1 around the rule-equivalent action (no calibration parameters),
  deterministic rule-based dispatch, env-v2 economics.
* One deterministic pass over all 98 reused test days
  (2026-02-01 → 2026-05-09). No reruns, no parameter changes after the
  read-out, regardless of outcome.
* Statistics: paired daily differences vs the *recorded* rule-based
  test series (`artifacts/test_evaluation.json`); moving-block bootstrap
  (block length 7, 10,000 draws); report mean, median, CI95,
  P(outperform), CVaR₁₀% regret, downside exposure, max daily loss, and
  the median info-gap vs the recorded info-MILP.
* The test split was examined by the previous phase: results are
  reported as a **reused-test confirmation**, not an untouched-test
  claim.

## 2026-07-18 — Phase 8 read-out and terminal state: `ENSEMBLE_SUCCESS`

One deterministic pass of the locked candidate over the 98 reused test
days, exactly as pre-registered:

* mean 46,767 / median 48,517 EUR/day (rule-based 46,850 / 48,464;
  info-MILP 47,599 / 49,294); median info-gap **+0.58%**.
* Paired vs rule-based: **−83 EUR/day, CI95 [−155, −30]**, CVaR₁₀%
  regret −569, downside exposure 141, **max daily loss 847 EUR**.
* Reliability vs the previous procedure on the same days: paired-CI
  width 125 EUR/day (was 5,752 pooled; the old validation-chosen seed
  landed at −6,114); worst daily loss 847 EUR (was five-figure); no
  seed selection anywhere in the loop.
* Honest boundary: the +27 EUR/day validation edge (P 91%) inverted to
  −83 under the winter→spring shift — the bound caps both tails; its
  insurance costs ~0.2% of revenue on this window.

Terminal state **ENSEMBLE_SUCCESS** (per the phase plan: no new
training; robustness materially improved; selection no longer depends
on one favorable seed; confirmed across temporal folds). Final report:
`reports/final_robust_rl_results.md`.

## 2026-07-18 — Phase 5 (retraining) formally pruned

ENSEMBLE_SUCCESS conditions are already met on validation by an
existing-policy construction: no new training is required. Specific
pruning reasons: (1) bounded-residual SAC duplicates what the post-hoc
Gate C already achieves architecturally, with headroom vs rule-based of
order 100 EUR/day against a multi-hour compute and code-skew risk;
(2) BC-initialized fine-tuning washed out twice in the previous phase;
(3) regime-aware retraining fails the Phase-4 evidence gate (n=1
regime days); (4) regret/CVaR-shaped rewards target a tail the gate has
already capped (max daily loss 1,430 EUR vs rule-based). Retraining
levers remain documented for a future phase with broader validation
coverage.
