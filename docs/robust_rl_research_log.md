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
