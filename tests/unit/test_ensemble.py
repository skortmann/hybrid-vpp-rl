"""Ensemble action aggregation and disagreement metrics."""

import numpy as np
import pytest

from hybrid_vpp.envs.ensemble import AGGREGATION_MODES, PolicyEnsemble, disagreement


class _Stub:
    """Policy stub returning a fixed action."""

    def __init__(self, action):
        self.action = np.asarray(action, dtype=float)

    def predict(self, obs, deterministic=True):
        assert deterministic
        return self.action, None


OBS = np.zeros(4)


def _ensemble(actions, **kwargs):
    return PolicyEnsemble([_Stub(a) for a in actions], **kwargs)


def test_mean_median_and_trimmed_aggregation():
    actions = [[0.0, -1.0], [0.5, 0.0], [1.0, 1.0], [0.5, 0.5], [0.0, 0.0]]
    mean_a, members = _ensemble(actions, mode="mean").predict(OBS)
    assert members.shape == (5, 2)
    assert mean_a == pytest.approx([0.4, 0.1])
    median_a, _ = _ensemble(actions, mode="median").predict(OBS)
    assert median_a == pytest.approx([0.5, 0.0])
    trimmed, _ = _ensemble(actions, mode="trimmed_mean", trim=1).predict(OBS)
    assert trimmed == pytest.approx([1 / 3, 1 / 6])  # extremes dropped per dim


def test_weighted_aggregation_and_validation():
    actions = [[1.0, 0.0], [0.0, 1.0]]
    weighted, _ = _ensemble(actions, mode="weighted", weights=[0.75, 0.25]).predict(OBS)
    assert weighted == pytest.approx([0.75, 0.25])
    with pytest.raises(ValueError):
        _ensemble(actions, mode="weighted")  # missing weights
    with pytest.raises(ValueError):
        _ensemble(actions, mode="weighted", weights=[0.9, 0.9])  # not a simplex
    with pytest.raises(ValueError):
        _ensemble(actions, mode="weighted", weights=[1.5, -0.5])  # negative


def test_aggregate_clips_to_action_box():
    # weighted combinations stay in the box even with extreme members
    agg = _ensemble([[1.0], [1.0], [1.0]], mode="mean").aggregate(np.array([[1.2], [1.4], [1.0]]))
    assert agg == pytest.approx([1.0])


def test_ensemble_constructor_guards():
    with pytest.raises(ValueError):
        _ensemble([[0.0]], mode="mean")  # single member
    with pytest.raises(ValueError):
        _ensemble([[0.0]] * 3, mode="nonsense")
    with pytest.raises(ValueError):
        _ensemble([[0.0]] * 4, mode="trimmed_mean", trim=2)  # nothing left
    assert set(AGGREGATION_MODES) == {"mean", "median", "trimmed_mean", "weighted"}


def test_disagreement_hand_computed():
    actions = np.array([[1.0, 0.0, 5.0], [-1.0, 0.0, -5.0]])
    d = disagreement(actions, active_dims=(0, 1))
    assert d["u"] == pytest.approx(1.0)  # each member 1.0 from center (0,0)
    assert d["max_pairwise"] == pytest.approx(2.0)
    assert d["per_dim_std"] == pytest.approx([1.0, 0.0])
    # unanimous members disagree by zero
    same = disagreement(np.array([[0.3, -0.2], [0.3, -0.2], [0.3, -0.2]]))
    assert same["u"] == pytest.approx(0.0, abs=1e-12)
    assert same["max_pairwise"] == pytest.approx(0.0, abs=1e-12)


def test_disagreement_masks_inactive_dims():
    # members differ wildly on an inactive dim only
    actions = np.array([[0.5, 1.0], [0.5, -1.0]])
    assert disagreement(actions, active_dims=(0,))["u"] == 0.0
    assert disagreement(actions)["u"] > 0.0
