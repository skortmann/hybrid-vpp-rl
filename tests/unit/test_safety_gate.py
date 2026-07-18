"""Safety gates: rule-equivalent action, gating semantics, calibration."""

import numpy as np
import pytest

from hybrid_vpp.envs.safety_gate import (
    SafetyGate,
    disagreement_thresholds,
    rule_equivalent_action,
)

RULE = rule_equivalent_action(gain_max=1.25)


def test_rule_equivalent_action_matches_pinned_semantics():
    # gain_max=1: identical to the raw vector pinned in test_strategic_mode
    assert rule_equivalent_action(1.0) == pytest.approx([2 / 1.2 - 1, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
    # interiorized gain range: gain 1.0 sits at 2/1.25 - 1 = 0.6
    assert RULE == pytest.approx([2 / 1.2 - 1, 1.0, 0.6, 0.6, 0.6, 0.0, 0.0])
    with pytest.raises(ValueError):
        rule_equivalent_action(0.5)


def test_gate_a_hard_fallback_on_disagreement():
    gate = SafetyGate("disagreement_threshold", RULE, u_thresholds={"DAA_GATE_CLOSURE": 0.1})
    rl = np.full(7, 0.9)
    action, rec = gate.apply(rl, "DAA_GATE_CLOSURE", u=0.05)
    assert action == pytest.approx(rl)
    assert rec["fallback"] is False and rec["alpha"] == 1.0
    action, rec = gate.apply(rl, "DAA_GATE_CLOSURE", u=0.2)
    assert action == pytest.approx(RULE)
    assert rec["fallback"] is True and rec["alpha"] == 0.0


def test_gate_b_scales_continuously_between_rule_and_rl():
    gate = SafetyGate("confidence_scaling", RULE, u_thresholds={"IDC_DECISION": 1.0})
    rl = RULE + 0.4
    action, rec = gate.apply(rl, "IDC_DECISION", u=0.0)
    assert rec["alpha"] == 1.0
    action, rec = gate.apply(rl, "IDC_DECISION", u=0.5)
    assert rec["alpha"] == pytest.approx(0.5)
    assert action == pytest.approx(np.clip(RULE + 0.2, -1, 1))
    action, rec = gate.apply(rl, "IDC_DECISION", u=2.0)
    assert rec["alpha"] == 0.0 and rec["fallback"] is True
    assert action == pytest.approx(RULE)


def test_gate_c_bounds_the_residual_per_dimension():
    bound = np.array([0.1, 0.1, 0.05, 0.05, 1.0, 1.0, 1.0])
    gate = SafetyGate("bounded_residual", RULE, max_residual=bound)
    rl = RULE + np.array([0.5, -0.5, 0.01, -0.5, 0.0, 0.0, 0.0])
    action, rec = gate.apply(rl, "IDA1_GATE_CLOSURE")
    expected = RULE + np.array([0.1, -0.1, 0.01, -0.05, 0.0, 0.0, 0.0])
    assert action == pytest.approx(np.clip(expected, -1, 1))
    assert rec["residual_clipped"] is True
    inside, rec = gate.apply(RULE + 0.01, "IDA1_GATE_CLOSURE")
    assert rec["residual_clipped"] is False


def test_gate_output_stays_in_action_box():
    gate = SafetyGate("bounded_residual", RULE, max_residual=5.0)
    action, _ = gate.apply(np.full(7, 3.0), "IDC_DECISION")
    assert (action <= 1.0).all() and (action >= -1.0).all()


def test_gate_constructor_guards():
    with pytest.raises(ValueError):
        SafetyGate("disagreement_threshold", RULE)  # missing thresholds
    with pytest.raises(ValueError):
        SafetyGate("confidence_scaling", RULE, u_thresholds={"X": 0.0})  # non-positive
    with pytest.raises(ValueError):
        SafetyGate("bounded_residual", RULE)  # missing bound
    with pytest.raises(ValueError):
        SafetyGate("bounded_residual", RULE, max_residual=-0.1)


def test_disagreement_thresholds_pool_by_event_type():
    cache = {
        "d1": {"DAA_GATE_CLOSURE": [0.1, 0.2], "IDC_DECISION": [0.0, 0.0]},
        "d2": {"DAA_GATE_CLOSURE": [0.3, 0.4], "IDC_DECISION": [1.0, 1.0]},
        "d3": {"DAA_GATE_CLOSURE": [99.0]},  # excluded: not in inner days
    }
    thr = disagreement_thresholds(cache, days=["d1", "d2"], quantile=0.5)
    assert thr["DAA_GATE_CLOSURE"] == pytest.approx(0.25)
    assert thr["IDC_DECISION"] == pytest.approx(0.5)
    # zero disagreement still yields a strictly positive threshold
    thr0 = disagreement_thresholds({"d": {"E": [0.0, 0.0]}}, days=["d"], quantile=0.9)
    assert thr0["E"] > 0
    with pytest.raises(ValueError):
        disagreement_thresholds(cache, days=["d1"], quantile=1.5)
