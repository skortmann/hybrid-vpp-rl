"""Deployment controller: ensemble → gate → bounded strategic action.

The operational wrapper around the promoted policy construction. Every
decision passes through, in order:

1. **schema check** at construction — the wrapper refuses to start if the
   configured action mode's schema version differs from the one the
   policies were trained under;
2. **missing-data fallback** — non-finite observations trigger the
   deterministic rule-equivalent action;
3. **out-of-distribution check** — observations outside the configured
   plausibility bounds trigger the fallback;
4. **safety gate** — the RL proposal is bounded toward the rule action;
5. **decision log** — every decision is recorded (observation digest,
   proposal, gated action, reason) so any trajectory can be replayed
   and audited.

The wrapper depends only on numpy, the loaded policies, and this
package's envs modules — no W&B and no training-only components.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from hybrid_vpp.envs.actions import ACTION_SCHEMA_VERSIONS
from hybrid_vpp.envs.ensemble import PolicyEnsemble, disagreement
from hybrid_vpp.envs.safety_gate import SafetyGate
from hybrid_vpp.envs.strategic import STRATEGIC_MASKS

TRAINED_SCHEMA = "act-v5-strategic"


def _digest(obs: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(obs, dtype=np.float32).tobytes()).hexdigest()[:16]


@dataclass
class DeploymentController:
    """Auditable, fallback-safe wrapper for the promoted RL construction."""

    ensemble: PolicyEnsemble
    gate: SafetyGate
    #: plausibility bounds for observations (config-derived, not fitted)
    obs_low: np.ndarray
    obs_high: np.ndarray
    #: identity of the deployed models, recorded in every decision log
    model_versions: tuple[str, ...] = ()
    action_mode: str = "strategic"
    decision_log: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        schema = ACTION_SCHEMA_VERSIONS.get(self.action_mode)
        if schema != TRAINED_SCHEMA:
            raise ValueError(
                f"schema mismatch: policies trained under {TRAINED_SCHEMA!r}, "
                f"configured action mode {self.action_mode!r} maps to {schema!r}"
            )
        self.obs_low = np.asarray(self.obs_low, dtype=float)
        self.obs_high = np.asarray(self.obs_high, dtype=float)
        if self.obs_low.shape != self.obs_high.shape or (self.obs_low > self.obs_high).any():
            raise ValueError("invalid observation bounds")

    def act(self, obs: np.ndarray, event_type: str) -> np.ndarray:
        """One gated decision; always returns a valid in-box action."""
        obs = np.asarray(obs, dtype=float)
        record: dict = {
            "event_type": event_type,
            "obs_digest": _digest(obs),
            "models": self.model_versions,
        }
        if obs.shape != self.obs_low.shape or not np.isfinite(obs).all():
            action = np.clip(self.gate.rule_action, -1.0, 1.0)
            record.update({"reason": "missing_data_fallback", "action": action.tolist()})
            self.decision_log.append(record)
            return action
        if ((obs < self.obs_low) | (obs > self.obs_high)).any():
            action = np.clip(self.gate.rule_action, -1.0, 1.0)
            record.update({"reason": "ood_fallback", "action": action.tolist()})
            self.decision_log.append(record)
            return action
        proposal, members = self.ensemble.predict(obs)
        active = {ev.name: dims for ev, dims in STRATEGIC_MASKS.items()}.get(event_type)
        u = disagreement(members, active)["u"]
        action, gate_record = self.gate.apply(proposal, event_type, u=u)
        record.update(
            {
                "reason": "gated_rl",
                "proposal": np.round(proposal, 5).tolist(),
                "action": np.round(action, 5).tolist(),
                "gate": gate_record,
            }
        )
        self.decision_log.append(record)
        return action
