"""Deployment controller: schema checks, fallbacks, decision logging."""

import numpy as np
import pytest

from hybrid_vpp.envs.deployment import DeploymentController
from hybrid_vpp.envs.ensemble import PolicyEnsemble
from hybrid_vpp.envs.safety_gate import SafetyGate, rule_equivalent_action

RULE = rule_equivalent_action(1.25)


class _Stub:
    def __init__(self, action):
        self.action = np.asarray(action, dtype=float)

    def predict(self, obs, deterministic=True):
        return self.action, None


def _controller(**kwargs):
    ensemble = PolicyEnsemble([_Stub(RULE + 0.3), _Stub(RULE + 0.1)], mode="mean")
    gate = SafetyGate("bounded_residual", RULE, max_residual=0.1)
    defaults = dict(
        ensemble=ensemble,
        gate=gate,
        obs_low=np.full(4, -10.0),
        obs_high=np.full(4, 10.0),
        model_versions=("seed0/best", "seed1/best"),
    )
    defaults.update(kwargs)
    return DeploymentController(**defaults)


def test_schema_mismatch_refuses_to_start():
    with pytest.raises(ValueError, match="schema mismatch"):
        _controller(action_mode="direct")
    with pytest.raises(ValueError, match="bounds"):
        _controller(obs_low=np.full(4, 1.0), obs_high=np.full(4, -1.0))


def test_normal_decision_is_gated_and_logged():
    c = _controller()
    action = c.act(np.zeros(4), "IDC_DECISION")
    # ensemble proposes RULE+0.2; residual bound 0.1 clips to RULE+0.1
    assert action == pytest.approx(np.clip(RULE + 0.1, -1, 1))
    rec = c.decision_log[-1]
    assert rec["reason"] == "gated_rl"
    assert rec["models"] == ("seed0/best", "seed1/best")
    assert "gate" in rec and rec["gate"]["event_type"] == "IDC_DECISION"
    assert len(rec["obs_digest"]) == 16


def test_missing_data_falls_back_to_rule_action():
    c = _controller()
    nan_obs = np.array([0.0, np.nan, 0.0, 0.0])
    action = c.act(nan_obs, "DAA_GATE_CLOSURE")
    assert action == pytest.approx(np.clip(RULE, -1, 1))
    assert c.decision_log[-1]["reason"] == "missing_data_fallback"
    # wrong shape is also treated as missing data
    action = c.act(np.zeros(3), "DAA_GATE_CLOSURE")
    assert c.decision_log[-1]["reason"] == "missing_data_fallback"


def test_out_of_distribution_falls_back():
    c = _controller()
    action = c.act(np.array([0.0, 11.0, 0.0, 0.0]), "IDA1_GATE_CLOSURE")
    assert action == pytest.approx(np.clip(RULE, -1, 1))
    assert c.decision_log[-1]["reason"] == "ood_fallback"


def test_decisions_are_replayable_from_the_log():
    c = _controller()
    obs = np.array([0.5, -0.5, 1.0, 2.0])
    first = c.act(obs, "IDC_DECISION")
    again = c.act(obs, "IDC_DECISION")
    assert first == pytest.approx(again)
    a, b = c.decision_log[-2:]
    assert a["obs_digest"] == b["obs_digest"]
    assert a["action"] == b["action"]
