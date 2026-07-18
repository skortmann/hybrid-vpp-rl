# Final robust RL results — selection, ensembles, and deployment reliability

**Terminal state: `ENSEMBLE_SUCCESS`** (2026-07-18, branch
`research/robust-rl-selection`; previous phase frozen at `rl-frontier-v1`)

## The question this phase answered

> Can the five existing SAC policies be selected, combined, or gated so
> that the controller generalizes more reliably than the current
> single-seed selection procedure?

**Yes — without any new training.** The promoted construction is the
deployment stack of the research plan: mean-action ensemble of the five
frozen eval-best checkpoints → per-dimension bounded residual (0.1)
around the rule-equivalent strategic action → deterministic rule-based
dispatch → feasibility projection. Every design choice (selection rule,
score coefficients, ensemble type, gate bound) was fixed on blocked
validation before the single pre-registered test read-out.

## Reused-test confirmation (98 days, 2026-02-01 → 2026-05-09)

The test split was examined by the previous phase; this is a
**reused-test confirmation**, not an untouched-test claim.

| Controller | Mean EUR/day | Median EUR/day |
|---|---|---|
| **Promoted: ensemble + bounded residual** | **46,767** | **48,517** |
| rule-based | 46,850 | 48,464 |
| information-equivalent MILP | 47,599 | 49,294 |
| perfect-foresight MILP | 52,007 | 52,573 |
| previous phase, 5 seeds pooled | 45,147 | 48,174 |
| previous phase, validation-chosen seed | 40,736 | 47,590 |

Paired vs rule-based (moving-block bootstrap, block 7): mean
**−83 EUR/day, CI95 [−155, −30]**, median regret −90, CVaR₁₀% regret
−569, downside exposure 141 EUR/day, **maximum daily loss 847 EUR**,
median info-gap **+0.58%**.

Reliability versus the previous procedure, same test days:

* Paired-CI width vs rule-based: **125 EUR/day** (was 5,752 for the
  pooled seeds; the old single-seed choice landed at −6,114).
* Worst-case daily loss vs rule-based: **847 EUR** (was five-figure).
* No seed selection at all — the failure mode that dominated the
  previous phase's residual gap is eliminated by construction.
* The construction beats the previous pooled-seed mean by +1,620 EUR/day
  and four of the five recorded individual seeds.

The honest boundary: the small positive validation edge (+27 EUR/day,
P(outperform) 91% on Nov–Jan) did not survive the winter→spring
distribution shift — on test the tight-bound variant sits 83 ± 60
EUR/day *below* rule-based (0.18% of daily revenue). The bound caps
losses and gains symmetrically; buying its insurance costs ~0.2% of
revenue on this test window.

## What was established on blocked validation (92 days, Nov–Jan)

1. **Selection is fixable (RQ1).** 6-block temporal validation with the
   pre-registered risk-adjusted score is the only rule family with
   positive held-out regret (+354 EUR/day LOBO); it ranks the
   pathological seed 0 last where the previous 30-day window ranked it
   first; Spearman(score, recorded test means) = +0.80 vs +0.40 for the
   raw validation mean. Early checkpoints generalize better (50k best,
   monotone-ish decline to 300k); seed effect std 517 ≈ checkpoint
   effect std 314 with interaction 403 — both selection layers mattered.
2. **Ensembles work (RQ2).** Every action-space ensemble beat every
   member; the plain mean ensemble turned mean regret positive (+124)
   and cut tail risk 5–8x (CVaR −1,822 vs −6.4k…−14.8k) and downside to
   309 EUR/day. Weighting by validation performance added nothing.
3. **Gates create a risk-return frontier (RQ3).** Bounded residual 0.1:
   downside 77 EUR/day, max loss 1,430, P(outperform) 91%; disagreement
   gate q80: P 93%. A random-fallback control recovered most of the
   disagreement gate's benefit — the safety value is bounded dampening
   toward the rule action, not the disagreement signal (day-level
   Spearman(u, regret) = +0.11, wrong sign; H3 refuted).
4. **Mechanisms identified, regime features pruned (RQ4/RQ5).** Failure
   days over-represent negative prices (lift 4.6–5.8, base rate 1%),
   low renewables (1.8–2.3), price volatility (1.3–2.0); reBAP
   volatility is largely neutralized by the hybrid architecture. The
   strongest regime trigger fires on 1 of 92 winter validation days —
   unvalidatable; regime conditioning formally pruned. Day-level
   market-feature OOD is a weak predictor (Spearman −0.05); recorded,
   not adopted.
5. **Stronger MILPs confirm the promotion (Phase 6).** Turnover-penalized
   and forecast-derated variants improve the classic info-MILP's tail
   but none approach the RL composites (best MILP CVaR −4,438 vs −437);
   the MILPs win medians (~58.2k) and lose hard days — the composite's
   advantage is robustness.
6. **Zero-shot robustness (RQ6).** The candidate stays ahead of
   rule-based under grid ±20%, BESS ×0.5/×2, and degraded forecasts;
   deviation profile pinned to rule-based (37.8 vs 37.6 MWh/day) so the
   penalty sweep 0–100 EUR/MWh preserves the ordering — except that at
   100 EUR/MWh the low-deviation info-MILP overtakes all trading-heavy
   controllers. Claims are calibrated to the 25 EUR/MWh regime.

## Pruned, with reasons

Retraining (bounded-residual SAC, regime-aware SAC, regret/CVaR
shaping): ENSEMBLE_SUCCESS conditions were met without it; the gate
already caps the tail architecturally; BC-style initialization washed
out twice in the previous phase; regime retraining fails the n=1
evidence gate. Critic-weighted ensembles: SAC Q-values are entropy- and
scale-shifted across seeds; calibration was not established. Regime
conditioning: see above.

## Deployment guidance

* **Deploy** `DeploymentController` (envs/deployment.py): schema check,
  missing-data fallback, OOD range fallback, gated ensemble, complete
  replayable decision logs. No W&B or training dependencies.
* **Risk posture is a dial, not a decision**: bound 0.1 ≈ rule-based
  ±0.2% revenue with hard loss caps; bound 0.4 / ungated ensemble
  trades tail width for upside (validation mean +124). Operators wanting
  strict compliance-style guarantees should run the tight bound;
  operators optimizing expectation should run the ensemble and accept
  ±5k daily swings vs rule-based.
* **Selection procedure for future retraining**: blocked temporal
  validation + risk-adjusted score + ensemble-all-seeds. Never eval-best
  single checkpoints on a short window.

## Remaining limitations

* Validation is winter-only; test is spring: no protocol on this dataset
  can validate seasonal transfer before deployment. The +0.58% median
  info-gap held; the mean edge did not.
* The test split is reused; a genuinely untouched claim needs new data
  (study window ends 2026-05-10).
* Negative-price regimes (the highest-lift failure mechanism) remain
  unaddressed — 1 validation day; first candidate for a phase with
  spring/summer validation data.

Full provenance: `artifacts/robust_selection/`, `artifacts/failure_days/`,
the research log (preserved at tag `robust-rl-final`), and the reports
under `reports/`; the research-time run registry was a local working
database and is not distributed.
