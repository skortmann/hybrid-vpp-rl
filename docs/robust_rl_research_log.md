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
