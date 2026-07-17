# RL research program

Goal: a controller that is demonstrably near-optimal on the hybrid-VPP
problem, measured against information-equivalent benchmarks on fixed
episodes. This page records the definition of success, the baseline that
research starts from, the formulation diagnosis, and the experiment
protocol. Results live in `experiments/registry.jsonl` (one JSON line per
run) and in the W&B project.

## Near-optimality definition

All controllers are evaluated on identical fixed validation/test episodes
(same days, profiles, prices, forecasts, constraints, costs). Two
references:

* **Information-equivalent MILP** — rolling optimization using only the
  forecasts available to the RL agent at the same decision times.
* **Perfect-foresight MILP** — realized prices/profiles; an unattainable
  information upper bound, reported but not the pass/fail target.

Primary criterion (per §evaluation): `Gap_info = (J_MILP_info − J_RL) /
max(|J_MILP_info|, ε)`; success requires

* median `Gap_info ≤ 5%` on the untouched test split,
* RL's lower 95% confidence bound ≥ best non-RL baseline,
* ≥ 5 independent seeds, no physical/accounting/leakage violations,
* trajectory audits showing no reward or settlement exploit.

If the information-equivalent MILP is weaker than the rule-based
controller on a given period, the stronger of the two is the practical
target. Screening anchors (first 30 validation days, EUR/day): do-nothing
47,908 · rule-based 50,861 · info-MILP 51,165 · perfect-foresight 54,994.

## Baseline record (v0.1.0)

`experiments/baseline_v1.json` snapshots the starting point: PPO on the
103-dimensional direct action space (act-v1), obs-v1 (523 dims), W&B runs
`nv2qwm96` (hot exploration, terminated) and `aesjpwo9` (2M-step run,
stopped at 500k). Verdict: evaluation oscillated between −157 and
+23 kEUR/day with no trend; action σ frozen at 0.37, clip fraction rising
to 0.35 at KL ≈ 0.03, explained variance ~0.6. Dominated by the rule-based
controller (~+56 kEUR/day) and stopped as uninformative.

## Formulation diagnosis (act-v1 checkpoint, 8 val days)

* **Event imbalance**: 768 dispatch steps vs 8 steps per auction gate per
  episode — market decisions receive ~1% of gradient samples.
* **Churn loop**: the policy accumulates ≈ +3.8M kEUR reward share at
  IDC/auction events and loses ≈ −5.8M at dispatch/settlement: it trades
  volume it cannot deliver, then pays imbalance. Credit assignment between
  a gate action and its settlement many steps later is the core difficulty.
* Feasibility projection rarely intervenes (0.1% of dispatch steps) — the
  physical layer is not the bottleneck; the market action semantics are.

## Action-space variants (schema-versioned)

| Schema | Mode | Dims (1-day) | Semantics |
|---|---|---|---|
| act-v1 | `direct` | 103 | signed incremental MW per QH slot |
| act-v2 | `target_position` | 103 | desired cumulative position per QH; translator trades the delta |
| act-v3 | `hourly_target` | 28 | hourly target-position anchors broadcast to QHs |
| act-v4 | `residual_hourly` | 28 | bounded correction (±`residual_scale_mw`) around the rule-based action; zero action ≡ rule-based (test-pinned) |

Target semantics make repeated identical intentions idempotent (no churn);
the residual variant starts at rule-based performance by construction.

## Experiment protocol

* Registry: `experiments/registry.jsonl` — id, phase, git commit, schema
  versions, algorithm, seed, budget, wall-clock, best-checkpoint validation
  KPIs on the fixed day set, W&B run name = experiment id.
* Phases: smoke → learnability (Level-1: DAA+BESS only, perfect forecasts —
  RL must approach the deterministic optimum) → screening (30 fixed val
  days) → tuning → multi-seed confirmation (full val split) → one locked
  test evaluation for the final candidate.
* Algorithms via a common adapter (`training/algorithms.py`): PPO,
  RecurrentPPO, SAC, TQC, TD3 — identical envs, callbacks, evaluation.
* Model selection always on true economic return (validation), never on
  training reward and never on test data.

## Advanced-algorithm program

Beyond the SB3 family, the program evaluates recent model-based,
offline-to-online, and projection-aware methods on the *compact*
environments (never first on the raw 103-dim tensor). Versions:
Python 3.12.9, torch 2.13 (CPU), SB3/SB3-Contrib 2.9.0, SBX 0.27 (JAX
0.10.2), Gymnasium 1.3.0, PyOptInterface 0.6.1 + Gurobi 13 / HiGHS.

### Capability matrix

| Method | On/Offline | Model-based | Recurrent | Cont. action | Dict obs | Hard-constraint integration | Demo support | Main risk |
|---|---|---|---|---|---|---|---|---|
| PPO (SB3) | online | no | no | yes | yes | via env projection | no | update instability at high KL |
| RecurrentPPO (contrib) | online | no | LSTM | yes | yes | via env projection | no | slow rollouts, state handling |
| SAC (SB3) | online | no | no | yes | yes | via env projection | prefill only | replay dominated by dispatch events |
| TQC (contrib) | online | no | no | yes | yes | via env projection | prefill only | quantile overfitting on spikes |
| TD3 (SB3) | online | no | no | yes | yes | via env projection | prefill only | brittle exploration |
| CrossQ (SBX) | online | no | no | yes | no (flat) | via env projection | prefill only | BatchNorm stats under event heterogeneity |
| RLPD (custom replay) | off→on | no | no | yes | flat | via env projection | **yes (core)** | prior-data imbalance |
| IQL / CQL (offline) | offline | no | no | yes | flat | via env projection | **yes (core)** | support mismatch after projection |
| TD-MPC2 (official impl) | online | **yes** | latent | yes | flat tensor | needs deterministic translator | optional | model error on rare gate events |
| DreamerV3 (official impl) | online | **yes** | RSSM | yes | vector | needs deterministic translator | no | compute cost; imagined constraint violations |
| Hybrid RL+MILP (H1–H6) | online | optimizer | no | low-dim | n/a | **exact (solver)** | n/a | solver latency, fallback design |

Not applicable: DQN/QR-DQN (no justified discretization yet); MaskablePPO
(no discrete sub-actions in current layouts).

### Compact environments (Tier 0)

* **Residual** (act-v4) — bounded corrections around rule-based ✓
* **Target-position** (act-v2/v3) ✓
* **Strategic** (act-v5) — 7 economic decision variables (DAA coverage,
  arbitrage scale, IDA/IDC correction gains, BESS tracking gain,
  curtailment price threshold, SoC bias) mapped by a deterministic
  translator that reuses the rule-based structure; mid-range parameters
  reproduce rule-based exactly (test-pinned). The critic sees only the
  pre-translation strategic action — the cleanest answer to projection
  aliasing (§8 of the extension brief).
* Prior-trajectory dataset (`training/datasets.py`): training-split days
  only, strategic schema, behaviors from do-nothing to high-turnover plus
  seeded random policies — the substrate for RLPD/IQL/CQL.

## The imbalance-speculation finding (env-v1 → env-v2)

The completed env-v1 screening produced scores far above the perfect-
foresight anchor (CrossQ+hourly: 61.4k EUR/day vs. 55.0k). Trajectory
audits showed why: under env-v1 economics, deliberate deviation between
contracted position and physical delivery settles at the historical reBAP
with **no penalty**, so target-position policies learned large-scale
imbalance speculation — 50–67% of their profit came from the imbalance
component, with |deviation| of the order of delivered energy and market
turnover at the volume caps. The accounting is correct and the behavior is
"legal" in-model, but it is not admissible as market-operation
performance: real balancing groups must not deviate deliberately, and the
price-taker assumption breaks at such volumes.

Consequences (all recorded in the registry decisions):

* env-v1 results from target-position formulations are **excluded from
  the operational ranking** (kept, annotated, and reported as a model
  finding). The apparent PPO-target "above the anchor" screening result
  did not replicate across seeds either (51.4k / 50.5k / 44.7k / 48.7k).
* **SAC-strategic is the clean env-v1 leader** (imbalance share −9%,
  deviation ratio 0.42, turnover ~660 MWh/day) — its action space
  structurally cannot speculate.
* **env-v2** adds `deviation_penalty_eur_per_mwh = 25` (a documented
  proxy for balancing-group compliance obligations) to the settlement for
  *all* controllers; the five leading formulations are re-screened under
  new experiment IDs (`V2-*`) against re-computed anchors.

## env-v2 confirmation wave (2026-07-16)

Multi-seed confirmation under penalized economics (300k steps, fixed 30
validation days; anchors: do-nothing 46,734 / rule-based 50,042 /
info-MILP 50,679 / perfect 54,704 EUR/day):

| formulation | seeds | mean | median range |
|---|---|---|---|
| CrossQ-strategic scratch | 3 | 45,794 (±0.7k) | 46.3–49.9k |
| CrossQ-strategic + replay prefill | 1 | 45,958 | 48,643 |
| SAC-strategic scratch | 3 | 44,249 (±0.4k) | 46.7–46.9k |
| SAC-strategic + replay prefill | 1 | 44,752 | 47,948 |

Findings: results are highly reproducible across seeds (spreads < 1.5k),
so this is a converged plateau, not noise; **replay prefill does not move
the plateau** (differences within seed noise) — prior transitions in the
buffer do not pull the entropy-regularized policy onto the rule-based
tight-tracking mode, although that mode lies inside the action space and
outperforms the learned policies by 4–6k. Near-optimality criteria are
NOT met under env-v2. Open levers before a frontier declaration:
behavior-cloned actor initialization (imitate, then fine-tune), offline
IQL/CQL on the prior dataset, hybrid MILP+RL, and the two queued
hypothesis tests (1M-step budget; colder-entropy SAC).

## Mechanism ladder (consolidated, 2026-07-17)

Three successively deeper causes of the RL–baseline gap were identified,
fixed or characterized, each with controlled evidence:

1. **Imbalance speculation** (env-v1): unpenalized reBAP settlement made
   deliberate deviation the dominant profit channel (up to 67% of profit).
   Fixed by env-v2 deviation penalty; contaminated results annotated.
2. **Corner-optimum unreachability** (act-v5): rule-based behavior sits at
   correction gains = 1 → raw action = +1, the corner of the squashed-
   Gaussian action box, which entropy-regularized policies cannot reach.
   Explains why prefill/BC/budget/entropy interventions all converged to
   the same sub-baseline attractor. Fixed by `strategic_gain_max = 1.25`
   (interior optimum); effect measurable but modest (medians +0.7k, best
   seed median within 1k of rule-based).
3. **Under-priced reBAP tail risk** (current frontier): paired per-day
   decomposition shows RL market revenue ≈ rule-based, but RL carries
   3–6× the deviation (150–195 vs 26–73 MWh/day); volatile-reBAP days
   produce five-figure losses (worst 5 of 30 days = 46% of the gap).
   The risk-neutral objective accepts this trade; a disciplined policy
   would not. Candidate remedies: risk-sensitive objectives (TQC/CVaR),
   deviation-penalty shaping during training, or hybrid dispatch.

Best honest results (env-v2, 30 fixed validation days, 3 seeds each):
cold-entropy SAC-strategic 46.4k ± 0.5k mean / ≈48.2k median; V4
interiorized 46.4k / 48.9k; rule-based 50.0k / 50.9k; info-MILP 50.7k.
Near-optimality criteria not met; the program continues on the risk-
discipline track.

## Status log

* 2026-07-15: baseline recorded and stopped; action variants act-v2/3/4
  implemented and test-pinned; Level-1 learnability (PPO/SAC) and Phase-2
  screening (PPO×4 action modes, SAC/TQC on hourly/residual) launched.
* 2026-07-15 (later): first screening results — SAC+hourly-target reaches
  48.9k EUR/day on the 30 fixed validation days (info-gap 4.5%);
  direct-PPO reference confirmed broken (6.9k). Advanced program started
  on branch `research/advanced-rl`: strategic env (act-v5), CrossQ via
  SBX, prior-trajectory dataset builder, capability matrix.
