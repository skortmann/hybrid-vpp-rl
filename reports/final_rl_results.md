# Final RL research results — hybrid VPP trading and dispatch

**Terminal state: `EMPIRICAL_FRONTIER_REACHED`** (2026-07-17)

## Winning formulation

SAC on the 7-dimensional strategic market action space (act-v5,
`strategic_gain_max = 1.25`) with **deterministic rule-based dispatch**
(`strategic_fixed_dispatch = true`), cold-entropy configuration
(`ent_coef 0.005`, UTD 2), 300k steps per seed, five seeds. Economics:
historical reBAP settlement + 25 EUR/MWh deviation penalty (env-v2, a
documented balancing-group-compliance proxy), identical for all
controllers.

## Test-split results (98 untouched days, 2026-02-01 → 2026-05-09)

| Controller | Mean EUR/day | Median EUR/day |
|---|---|---|
| RL hybrid (5 seeds pooled) | 45,147 | 48,174 |
| — seed range | 40,736 – 48,104 | 47,590 – 48,660 |
| do-nothing | 42,512 | 41,462 |
| rule-based | 46,850 | 48,464 |
| information-equivalent MILP | 47,599 | 49,294 |
| perfect-foresight MILP | 52,007 | 52,573 |

* Median information-equivalent gap: **+0.07%** (criterion ≤ 5%: **met**)
* Median perfect-foresight gap: +8.2% (information bound, reported only)
* Paired RL − rule-based: −1,703 EUR/day, 95% CI **[−5,085, +667]** —
  statistically **indistinguishable** from the strongest non-perfect
  baseline; not better.
* Seed-mean dispersion on test: 2,680 EUR/day — **seed consistency
  criterion not met**; the validation-best seed (50.6k on 30 validation
  days) was the *worst* test generalizer (40.7k), i.e. 30 validation days
  over-select and generalization variance dominates the remaining gap.
* Best single seed (seed 2): 48,104 mean — **above** rule-based and within
  1% of the info-MILP on test; unusable as a claim without a selection
  protocol that identifies it ex ante.

Because RL is indistinguishable from — but not demonstrably better than —
the strongest feasible baseline, and the five-seed consistency requirement
fails, `SUCCESS_NEAR_OPTIMAL` is **not** declared.

## The mechanism ladder (what closed the gap, with evidence)

1. **Imbalance speculation** (env-v1): unpenalized reBAP deviation was the
   dominant profit channel (50–67% of profit; one policy exceeded perfect
   foresight by 12%). Fixed by penalized economics; contaminated results
   annotated, never deleted.
2. **Corner-optimum unreachability**: rule-based behavior mapped to the
   corner of the squashed-Gaussian action box; entropy-regularized
   policies structurally cannot reach it. Fixed by interiorizing the gain
   range (+0.7k median).
3. **Under-priced reBAP tail risk**: policies carried 3–6× rule-based's
   deviation; volatile-reBAP days produced five-figure paired losses
   (worst 5 of 30 days = 46% of the gap). Fixed architecturally by
   deterministic dispatch (+3.5k mean) — the single largest improvement.
4. **Residual (open)**: −1.7k mean vs rule-based, CI spanning zero, driven
   by seed-level generalization variance, not by any identified
   formulation defect.

## Tested and insufficient (all multi-seed, registry-recorded)

Direct 103-dim actions (catastrophic churn); target-position semantics
(idempotent but speculation-prone under env-v1, weak under env-v2); PPO in
all variants; TQC; replay prefill (RLPD-style; Δ within noise); behavior-
cloned initialization (twice — washes out during fine-tuning); doubled
training budget (plateau unmoved; 1M run pruned as mooted); risk shaping
(train 50 / evaluate 25: no gain). Formally pruned without full runs, with
mechanism-based justification: offline IQL/CQL (the BC arm already
started from the dataset's best policy and lost it) and TD-MPC2/DreamerV3
(sample efficiency and credit assignment are not the binding constraint —
300k-step runs converge with sub-1k seed spreads).

## Recommended next research steps

1. Seed/checkpoint selection on a larger validation set (the 30-day set
   over-selects; use the full 92 validation days or cross-validation).
2. Ensemble or seed-averaged policies to attack generalization variance.
3. Upside-focused objectives: RL's medians consistently exceed baselines —
   the value concentrates in high-variance days; a formulation that keeps
   rule-based's floor while retaining that upside (e.g. constrained or
   regret-based fine-tuning) is the most promising direction.
4. Regime-conditioned policies (season/volatility features) for the
   February–May distribution shift observed here.

Full provenance: `runs/research_state.sqlite` (46 experiments, every
state transition and decision), `experiments/registry.jsonl`,
`docs/rl_research_log.md`, W&B project `hybrid-vpp-rl`.
