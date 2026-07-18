# Controllers

The public controller set, from `hybrid_vpp.controllers`:

| Controller | Role | Interface |
|---|---|---|
| `DoNothingController` | passive floor (sells nothing, delivers as-available) | `act(event, sim)` |
| `RuleBasedController` | deterministic operational **reference and fallback** | `act(event, sim)` |
| `OptimizationController` | information-equivalent MILP **benchmark**; perfect-foresight configuration gives the upper bound | `act(event, sim)` |
| `EnsembleDeploymentController` | **recommended deployment design** | `act(obs, event_type)` |

Simulator-facing controllers plug into `controllers.base.run_episode`;
the deployment controller consumes Gymnasium observations from
`envs.hybrid_vpp_env.HybridVppEnv`.

## Rule-based controller

Sells the renewable forecast day-ahead (coverage 1.0), adds a
battery-arbitrage schedule, refreshes positions at each intraday auction
and at continuous-intraday decisions, and tracks the contracted position
at dispatch time with the battery. Conservative by construction; its
deviation volume defines the compliance envelope the deployment design
inherits.

## Optimization controller (benchmark)

Rolling-horizon MILP (PyOptInterface; HiGHS by default, Gurobi if
licensed) re-optimizing at every gate on exactly the forecasts available
to the RL agent. Optional variants: a turnover penalty on schedule churn
across successive re-optimizations and a renewable-forecast derate
(conservative planning). Requires the `optimization` extra.

## Strategic RL policies

Trained policies act on the strategic action space (schema `act-v5`):
seven economic decision variables translated deterministically through
the rule-based structure —

| dim | meaning | active at |
|---|---|---|
| 0 | day-ahead renewable coverage (0–1.2) | DAA gate |
| 1 | day-ahead arbitrage scale (0–1) | DAA gate |
| 2 | intraday-auction correction gain (0–1.25) | IDA1/2/3 gates |
| 3 | continuous-intraday correction gain (0–1.25) | IDC decisions |
| 4–6 | dispatch tracking gain, curtailment threshold, SoC bias | physical dispatch |

In the promoted configuration (`strategic_fixed_dispatch: true`) physical
dispatch is delegated to the deterministic rule-based tracker and
**dimensions 4–6 are inert**. The schema deliberately stays
seven-dimensional: the released checkpoints were trained on it, and a
narrower schema without trained models would serve no one. The
rule-equivalent action (the point of the box that reproduces the
rule-based controller exactly, pinned by tests) is available as
`rule_equivalent_action(gain_max)`.

## Ensemble deployment controller (recommended)

The promoted design of the study:

```
a_deploy = a_rule + clip( mean_i a_i  −  a_rule,  −0.1, +0.1 )
```

— the mean strategic action of the five released SAC policies, clipped
per dimension to a bounded residual around the rule-equivalent action,
with deterministic rule-based dispatch and exact feasibility projection
downstream. The wrapper adds, in order: an action-schema check at
construction, a missing-data fallback (non-finite observation → rule
action), an out-of-range fallback, the safety gate, and a replayable
decision log (observation digest, member proposal, gated action, reason,
model identities).

```python
import numpy as np
from hybrid_vpp.controllers import (
    EnsembleDeploymentController, PolicyEnsemble, SafetyGate,
    rule_equivalent_action,
)
from hybrid_vpp.training.algorithms import algo_class

models = [algo_class("sac").load(p) for p in checkpoint_paths]
rule = rule_equivalent_action(gain_max=1.25)
controller = EnsembleDeploymentController(
    ensemble=PolicyEnsemble(models, mode="mean"),
    gate=SafetyGate("bounded_residual", rule, max_residual=0.1),
    obs_low=obs_low, obs_high=obs_high,
    model_versions=tuple(p.name for p in checkpoint_paths),
)
action = controller.act(obs, event_type)   # always a valid in-box action
```

Why an ensemble instead of the best trained seed: seed selection from a
validation window chose the *worst* test generalizer in this study, while
the ensemble beats every individual member on validation mean revenue and
narrows the paired test confidence interval 46-fold. The residual bound
is a risk dial: 0.1 gives rule-based ±0.2% revenue with a hard cap on
daily losses (847 EUR worst observed vs rule-based on 98 test days);
larger bounds (or the ungated ensemble) trade tail width for expectation.

The runnable end-to-end example is `examples/run_deployment_controller.py`;
model files ship as release assets (see [reproducibility](reproducibility.md)).
