# Robust RL research plan — selection, ensembles, and deployment reliability

Phase 2 of the RL research program. The previous phase terminated in
`EMPIRICAL_FRONTIER_REACHED` (tag `rl-frontier-v1`, commit `17b002e`): the
V6 hybrid controller (SAC on act-v5 strategic actions, deterministic
rule-based dispatch) reaches a ~0.07% median information-equivalent gap on
the test split but is statistically indistinguishable from rule-based on
means, with seed-level generalization variance as the dominant unresolved
issue — the validation-best seed was the worst test generalizer.

**This phase is validation research.** The study window ends 2026-05-10;
no unused later data exist, so there is no new untouched holdout. The
98-day test split was examined by the previous phase and is *reused*, not
untouched. It is touched exactly once more: a final locked confirmation of
the single promoted controller in Phase 8, reported explicitly as a
reused-test confirmation.

## Research questions

* **RQ1** — Can a blocked temporal validation protocol reliably identify
  the SAC seed/checkpoint that generalizes best?
* **RQ2** — Do ensembles of the existing five SAC policies reduce seed
  variance?
* **RQ3** — Can a safety gate retain rule-based downside protection while
  preserving RL upside?
* **RQ4** — Which observable regimes explain RL underperformance?
* **RQ5** — Does regime conditioning or regret-aware training improve
  robustness?
* **RQ6** — Is the result stable under changes to deviation penalty,
  asset sizing, grid limits, and forecast quality?

## Hypotheses

* **H1**: the previous 30-day (November-only) validation window
  over-selected; fold-diverse selection (mean/worst-fold/risk-adjusted
  over 6 blocks spanning Nov–Jan) picks seeds whose ranking transfers.
* **H2**: action-space ensembling reduces the seed-mean dispersion
  (2,680 EUR/day on test) substantially, because per-seed errors are
  partly idiosyncratic.
* **H3**: policy disagreement is elevated on days where RL loses to
  rule-based, making a disagreement gate a usable safety signal.
* **H4**: RL underperformance concentrates in identifiable regimes
  (reBAP volatility, forecast-revision magnitude, negative-price
  clusters) rather than being uniform.
* **H5**: bounded-residual or regret-aware retraining only helps if
  Phases 1–3 leave a mechanism-attributable gap; otherwise it is pruned.

## Data and evaluation hierarchy

```
train 2024-06-14 → 2025-11-01   (policy training; frozen for Phases 1–4)
validation 2025-11-01 → 2026-02-01 (92 days) — all method development
    inner: leave-one-block-out over contiguous blocks (selection rules)
    outer: held-out blocks (assessment of the selection procedure itself)
test 2026-02-01 → 2026-05-09 (98 days, REUSED) — one locked Phase-8
    confirmation of the single promoted controller; never used during
    development; never called untouched
```

Blocked temporal validation: the 92 validation days are split into
contiguous, non-overlapping blocks (default 6 × ~15 days; configurable),
no shuffling, serial/seasonal structure preserved. Selection-rule
reliability (RQ1) is measured by leave-one-block-out: apply each rule to
5 blocks, score the chosen policy on the held-out block, repeat over all
blocks. The recorded per-day test results of the five `best` checkpoints
(artifacts/test_evaluation.json, already burned by the previous phase)
may be read **after** the selection rule is locked, as a read-only
consistency check in the report — never as a criterion for choosing the
rule.

## Experiment phases, budgets, stopping

* **Phase 0 — reproducibility** (done before this plan was written):
  full test suite green; five final checkpoints reproduce recorded test
  revenues to the cent; tag `rl-frontier-v1`; this branch.
* **Phase 1 — selection analysis** (~3,300 episodes ≈ hours, CPU):
  evaluate 5 seeds × 7 checkpoints (50k…300k, best) on all 92 validation
  days; baselines (do-nothing, rule-based, info-MILP) on the same days;
  fold metrics (mean, median, std, worst-fold, CVaR₁₀%, deviation cost,
  turnover, cycles, regret vs rule-based and vs info-MILP); predefined
  selection rules: highest mean / highest median / highest worst-fold /
  risk-adjusted score / lowest negative regret / Pareto. Risk-adjusted
  score (all λ configurable, defaults fixed here before any evaluation):
  S_i = mean_i − 0.5·σ_i(folds) − 0.5·CVaR₁₀%ᵢ(daily regret) − 0.25·R_i.
* **Phase 2 — ensembles** (~600 episodes): mean-, median-, trimmed-mean-,
  validation-weighted-action ensembles of the five best checkpoints (and
  the per-rule selected checkpoints); disagreement metrics logged per
  step; critic-weighted variant only if Q-calibration passes.
* **Phase 3 — safety gates** (~1,000 episodes): Gate A (disagreement
  threshold), Gate B (continuous confidence scaling), Gate C (bounded
  residual around the rule-equivalent strategic action), Gate E (regime
  fallback, only after Phase 4). Thresholds chosen on inner folds only.
* **Phase 4 — failure-day and regime analysis**: automated worst-day
  diagnostics for every controller family; mechanism clustering; regime
  features added only with evidence + leakage tests.
* **Phase 5 — retraining** (conditional on Phases 1–4; ~6–10 runs ×
  300k steps): bounded-residual SAC, regime-features SAC, regret/CVaR
  shaping. Registry namespace `robust_retrain`, supervised by the
  existing research supervisor.
* **Phase 6 — benchmark strengthening**: turnover-penalized,
  deviation-risk-aware, robust, and (if tractable) scenario-based MILP
  variants under identical information.
* **Phase 7 — sensitivity**: deviation penalty {0,10,25,50,100},
  grid/BESS/composition scaling, forecast-quality degradation; zero-shot
  vs retrained clearly separated.
* **Phase 8 — final confirmation**: ≥10 seeds of the locked candidate if
  retraining was justified (else the locked existing-policy construction);
  hierarchical bootstrap (seeds × day-blocks); one reused-test
  confirmation.

Stopping: `ROBUST_SUCCESS`, `ENSEMBLE_SUCCESS`, or
`EMPIRICAL_ROBUSTNESS_FRONTIER` as defined in the phase directive;
recorded in the research registry under the namespace `robust`.

## Promotion criteria

A candidate replaces the current single-seed procedure only with: equal
or higher mean and median blocked-validation revenue, lower fold/seed
dispersion, lower negative-regret CVaR, P(day beats rule-based) trending
toward >75%, no new physical or accounting violations, and inference
cheap enough for the decision cadence (<1 s per gate decision). No
selection rule may be revised after the Phase-8 read-out.

## Risks and failure modes

* **Winter-only validation**: all 92 validation days are Nov–Jan; test is
  Feb–May. Fold structure captures within-winter variation only —
  seasonal transfer remains unverifiable before the locked confirmation.
  Mitigation: regime features (Phase 4) use season-invariant drivers
  (volatility, forecast error) rather than calendar identity.
* **Selection-rule overfitting**: six rules on six blocks invites rule
  shopping. Mitigation: LOBO protocol fixed here; rule locked before the
  test read-out; the read-out is reported for all rules, not only the
  winner.
* **Ensemble action non-sense**: averaging squashed actions can leave the
  data manifold. Mitigation: aggregation in raw pre-translator action
  space with bounds preserved; per-dimension disagreement audited.
* **Critic miscalibration** (Gate D): SAC Q-values are entropy-shifted
  and scale-shifted across seeds. Gate D is skipped unless calibration
  passes on validation folds.
* **Compute**: checkpoint matrix dominates (~3.3k episodes). Runs are
  process-parallel over checkpoints; artifacts cached per
  (checkpoint, day) so reruns are incremental.
* **Code skew**: no edits to `envs/`, `sim/`, or `training/` while
  long-running evaluation jobs are live (documented hazard from the
  previous phase).

## Expected artifacts

`artifacts/robust_selection/` (checkpoint_matrix.{csv,json},
fold_results.csv, ensemble_results.json, disagreement_analysis.json,
safety_gate_results.json, sensitivity_results.json),
`reports/checkpoint_selection_analysis.md`, `reports/ensemble_analysis.md`,
`reports/failure_day_analysis.md`, `reports/regime_analysis.md`,
`reports/robust_milp_analysis.md`, `reports/final_robust_rl_results.{md,json}`,
plots under `docs/assets/robust/`, and a running
`docs/robust_rl_research_log.md`.
