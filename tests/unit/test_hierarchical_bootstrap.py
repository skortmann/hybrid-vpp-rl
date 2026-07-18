"""Hierarchical bootstrap: block sampling, degenerate cases, seed variance."""

import numpy as np
import pandas as pd
import pytest

from hybrid_vpp.evaluation.hierarchical_bootstrap import (
    _moving_block_indices,
    hierarchical_bootstrap,
)

DAYS = pd.date_range("2026-02-01", periods=40, freq="D")


def test_block_indices_are_contiguous_in_bounds_and_trimmed():
    rng = np.random.default_rng(0)
    idx = _moving_block_indices(40, 7, rng)
    assert len(idx) == 40
    assert idx.min() >= 0 and idx.max() < 40
    # each 7-run is contiguous
    for start in range(0, 35, 7):
        run = idx[start : start + 7]
        assert (np.diff(run) == 1).all()
    with pytest.raises(ValueError):
        _moving_block_indices(10, 0, rng)
    with pytest.raises(ValueError):
        _moving_block_indices(10, 11, rng)


def test_constant_difference_collapses_the_interval():
    per_day = pd.DataFrame({f"s{i}": np.full(40, 110.0) for i in range(3)}, index=DAYS)
    reference = pd.Series(100.0, index=DAYS)
    r = hierarchical_bootstrap(per_day, reference, n_boot=200, block_len=5)
    assert r["mean_diff"] == pytest.approx(10.0)
    assert r["ci95_low"] == pytest.approx(10.0)
    assert r["ci95_high"] == pytest.approx(10.0)
    assert r["p_outperform"] == 1.0
    assert r["seed_std"] == 0.0


def test_seed_heterogeneity_widens_the_interval():
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 1, size=(40, 4))
    homogeneous = pd.DataFrame(100.0 + noise, index=DAYS, columns=[f"s{i}" for i in range(4)])
    offsets = np.array([-30.0, -10.0, 10.0, 30.0])
    heterogeneous = homogeneous + offsets
    reference = pd.Series(95.0, index=DAYS)
    r_homo = hierarchical_bootstrap(homogeneous, reference, n_boot=500, block_len=5, seed=2)
    r_hetero = hierarchical_bootstrap(heterogeneous, reference, n_boot=500, block_len=5, seed=2)
    width_homo = r_homo["ci95_high"] - r_homo["ci95_low"]
    width_hetero = r_hetero["ci95_high"] - r_hetero["ci95_low"]
    assert width_hetero > 3 * width_homo
    assert r_hetero["seed_std"] > 10 * r_homo["seed_std"]


def test_reference_coverage_is_enforced_and_result_is_deterministic():
    per_day = pd.DataFrame({"s0": np.arange(40.0), "s1": np.arange(40.0)}, index=DAYS)
    with pytest.raises(ValueError):
        hierarchical_bootstrap(per_day, pd.Series(0.0, index=DAYS[:10]))
    reference = pd.Series(10.0, index=DAYS)
    a = hierarchical_bootstrap(per_day, reference, n_boot=100, seed=42)
    b = hierarchical_bootstrap(per_day, reference, n_boot=100, seed=42)
    assert a == b
