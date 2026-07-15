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

## Status log

* 2026-07-15: baseline recorded and stopped; action variants act-v2/3/4
  implemented and test-pinned; Level-1 learnability (PPO/SAC) and Phase-2
  screening (PPO×4 action modes, SAC/TQC on hourly/residual) launched.
* 2026-07-15 (later): first screening results — SAC+hourly-target reaches
  48.9k EUR/day on the 30 fixed validation days (info-gap 4.5%);
  direct-PPO reference confirmed broken (6.9k). Advanced program started
  on branch `research/advanced-rl`: strategic env (act-v5), CrossQ via
  SBX, prior-trajectory dataset builder, capability matrix.
