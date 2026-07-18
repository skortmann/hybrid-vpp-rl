"""Public controller API.

Baselines and benchmarks (simulator-facing, ``controller.act(event, sim)``):

* :class:`DoNothingController` — sells nothing; the passive floor.
* :class:`RuleBasedController` — the deterministic operational reference;
  fallback anchor of the deployment design.
* :class:`OptimizationController` — the information-equivalent MILP
  benchmark (rolling re-optimization on the forecasts available at each
  gate; perfect-foresight configuration available for the upper bound).

Deployment (observation-facing, ``controller.act(obs, event_type)``):

* :class:`EnsembleDeploymentController` — the promoted design: mean
  strategic action of the released SAC policies, bounded residual around
  the rule-equivalent action, deterministic rule-based dispatch,
  missing-data/out-of-range fallbacks, and a replayable decision log.
  Built from :class:`~hybrid_vpp.envs.ensemble.PolicyEnsemble` and
  :class:`~hybrid_vpp.envs.safety_gate.SafetyGate`.

Importing :class:`OptimizationController` requires the ``optimization``
extra (PyOptInterface); it is therefore exported lazily.
"""

from hybrid_vpp.controllers.rule_based import RuleBasedController
from hybrid_vpp.controllers.simple import DoNothingController
from hybrid_vpp.envs.deployment import DeploymentController as EnsembleDeploymentController
from hybrid_vpp.envs.ensemble import PolicyEnsemble
from hybrid_vpp.envs.safety_gate import SafetyGate, rule_equivalent_action

__all__ = [
    "DoNothingController",
    "EnsembleDeploymentController",
    "OptimizationController",
    "PolicyEnsemble",
    "RuleBasedController",
    "SafetyGate",
    "rule_equivalent_action",
]


def __getattr__(name: str):
    if name == "OptimizationController":
        from hybrid_vpp.controllers.optimization import OptimizationController

        return OptimizationController
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
