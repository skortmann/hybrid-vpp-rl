"""Safety gates between RL strategic actions and the rule-based fallback.

A gate decides, at decision time only, how much of the RL (or ensemble)
action reaches the market — falling back to the rule-equivalent
strategic action when the policy signal is untrustworthy. Gates never
see realized profits, future prices, or future deviations; the only
inputs are the proposed action, the event type, and the ensemble
disagreement computed from the same observation.

Gate A (``disagreement_threshold``): hard fallback to the rule action
when member disagreement exceeds a per-event threshold.
Gate B (``confidence_scaling``): continuous interpolation
``rule + alpha * (rl - rule)`` with ``alpha`` shrinking in disagreement.
Gate C (``bounded_residual``): clip the deviation from the rule action
to a per-dimension bound.

Thresholds and bounds must be calibrated on inner validation blocks
(:func:`disagreement_thresholds`), never on held-out data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

#: raw action reproducing the rule-based controller under the strategic
#: translator (see the pinned test in tests/unit/test_strategic_mode.py):
#: DAA coverage 1.0, full arbitrage, IDA/IDC/tracking gain 1.0, curtail
#: threshold 0 EUR/MWh, no bias.


def rule_equivalent_action(gain_max: float = 1.0) -> np.ndarray:
    """Raw 7-dim action equivalent to the rule-based controller."""
    if gain_max < 1.0:
        raise ValueError("gain_max must be >= 1")
    gain_raw = 2.0 / gain_max - 1.0
    return np.array([2.0 / 1.2 - 1.0, 1.0, gain_raw, gain_raw, gain_raw, 0.0, 0.0])


GateMode = Literal["disagreement_threshold", "confidence_scaling", "bounded_residual"]


@dataclass
class SafetyGate:
    """Combine an RL proposal with the rule-based fallback action."""

    mode: GateMode
    rule_action: np.ndarray
    #: per-event disagreement scale: hard cutoff (Gate A) or the
    #: disagreement at which alpha reaches zero (Gate B)
    u_thresholds: dict[str, float] | None = None
    #: per-dimension residual bound (Gate C); scalar broadcasts
    max_residual: np.ndarray | float | None = None

    def __post_init__(self) -> None:
        self.rule_action = np.asarray(self.rule_action, dtype=float)
        if self.mode in ("disagreement_threshold", "confidence_scaling"):
            if not self.u_thresholds:
                raise ValueError(f"{self.mode} requires u_thresholds")
            if any(t <= 0 for t in self.u_thresholds.values()):
                raise ValueError("u_thresholds must be positive")
        if self.mode == "bounded_residual":
            if self.max_residual is None:
                raise ValueError("bounded_residual requires max_residual")
            bound = np.broadcast_to(
                np.asarray(self.max_residual, dtype=float), self.rule_action.shape
            )
            if (bound < 0).any():
                raise ValueError("max_residual must be non-negative")
            self.max_residual = np.array(bound)

    def apply(
        self, rl_action: np.ndarray, event_type: str, u: float = 0.0
    ) -> tuple[np.ndarray, dict]:
        """Gated action plus a decision record for the audit log."""
        rl = np.asarray(rl_action, dtype=float)
        if self.mode == "disagreement_threshold":
            threshold = self.u_thresholds[event_type]
            fallback = u > threshold
            action = self.rule_action if fallback else rl
            record = {"alpha": 0.0 if fallback else 1.0, "fallback": bool(fallback)}
        elif self.mode == "confidence_scaling":
            threshold = self.u_thresholds[event_type]
            alpha = float(np.clip(1.0 - u / threshold, 0.0, 1.0))
            action = self.rule_action + alpha * (rl - self.rule_action)
            record = {"alpha": alpha, "fallback": alpha == 0.0}
        else:  # bounded_residual
            residual = np.clip(rl - self.rule_action, -self.max_residual, self.max_residual)
            action = self.rule_action + residual
            clipped = bool((np.abs(rl - self.rule_action) > self.max_residual).any())
            record = {"alpha": 1.0, "fallback": False, "residual_clipped": clipped}
        record.update({"u": float(u), "event_type": event_type})
        return np.clip(action, -1.0, 1.0), record


def disagreement_thresholds(
    disagreement_by_day: dict[str, dict[str, list[float]]],
    days: list[str],
    quantile: float = 0.8,
) -> dict[str, float]:
    """Per-event-type disagreement quantiles over the given (inner) days.

    ``disagreement_by_day`` maps day -> event type -> per-step ``u``
    values, as cached by :mod:`hybrid_vpp.evaluation.ensemble_eval`.
    """
    if not 0 < quantile < 1:
        raise ValueError("quantile must be in (0, 1)")
    pooled: dict[str, list[float]] = {}
    for day in days:
        for ev, values in disagreement_by_day[day].items():
            pooled.setdefault(ev, []).extend(values)
    out = {}
    for ev, values in pooled.items():
        q = float(np.quantile(np.asarray(values), quantile))
        out[ev] = max(q, 1e-9)  # keep thresholds strictly positive
    return out
