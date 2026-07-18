"""Action-space ensembles of trained policies and disagreement metrics.

An ensemble aggregates the deterministic raw actions of N member
policies per step — in the raw (pre-translator) action space, where every
dimension lives in [-1, 1] and the strategic translator is monotone per
dimension. Disagreement between members is computed on the dims that are
active for the current market event (:data:`STRATEGIC_MASKS`), since
inactive dims never reach the simulator.

Deployment note: this module depends only on numpy and the loaded
policies' ``predict`` interface — no W&B or training-only components.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

AGGREGATION_MODES = ("mean", "median", "trimmed_mean", "weighted")


@dataclass
class PolicyEnsemble:
    """Aggregate deterministic member actions into one raw action."""

    models: list
    mode: str = "mean"
    weights: np.ndarray | None = None  # required for mode="weighted"
    trim: int = 1  # members dropped per extreme for trimmed_mean

    def __post_init__(self) -> None:
        if self.mode not in AGGREGATION_MODES:
            raise ValueError(f"mode must be one of {AGGREGATION_MODES}, got {self.mode!r}")
        if len(self.models) < 2:
            raise ValueError("an ensemble needs at least two member policies")
        if self.mode == "weighted":
            if self.weights is None:
                raise ValueError("mode='weighted' requires weights")
            w = np.asarray(self.weights, dtype=float)
            if w.shape != (len(self.models),) or (w < 0).any() or not np.isclose(w.sum(), 1.0):
                raise ValueError("weights must be a non-negative simplex vector, one per member")
            self.weights = w
        if self.mode == "trimmed_mean" and len(self.models) <= 2 * self.trim:
            raise ValueError("trimmed_mean needs more than 2*trim members")

    def member_actions(self, obs: np.ndarray) -> np.ndarray:
        """(n_members, action_dim) deterministic raw actions."""
        return np.stack(
            [np.asarray(m.predict(obs, deterministic=True)[0], dtype=float) for m in self.models]
        )

    def aggregate(self, actions: np.ndarray) -> np.ndarray:
        """Combine member actions (rows) into one action, clipped to [-1, 1]."""
        if self.mode == "mean":
            agg = actions.mean(axis=0)
        elif self.mode == "median":
            agg = np.median(actions, axis=0)
        elif self.mode == "trimmed_mean":
            ordered = np.sort(actions, axis=0)
            agg = ordered[self.trim : actions.shape[0] - self.trim].mean(axis=0)
        else:  # weighted
            agg = self.weights @ actions
        return np.clip(agg, -1.0, 1.0)

    def predict(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(aggregated action, member actions) for one observation."""
        actions = self.member_actions(obs)
        return self.aggregate(actions), actions


# ---------------------------------------------------------------- disagreement


def disagreement(actions: np.ndarray, active_dims: tuple[int, ...] | None = None) -> dict:
    """Ensemble disagreement over the dims that matter for this event.

    ``actions`` is (n_members, action_dim). Returns the mean squared
    distance to the ensemble mean (``u``), the maximum pairwise L2
    distance, and per-dimension standard deviations.
    """
    a = np.asarray(actions, dtype=float)
    if active_dims is not None:
        a = a[:, list(active_dims)]
    center = a.mean(axis=0)
    u = float(((a - center) ** 2).sum(axis=1).mean())
    max_pairwise = 0.0
    for i in range(len(a)):
        d = np.linalg.norm(a[i + 1 :] - a[i], axis=1)
        if d.size:
            max_pairwise = max(max_pairwise, float(d.max()))
    return {
        "u": u,
        "max_pairwise": max_pairwise,
        "per_dim_std": a.std(axis=0, ddof=0).tolist(),
    }
