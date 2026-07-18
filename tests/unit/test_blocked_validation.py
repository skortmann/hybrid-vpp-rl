"""Blocked temporal validation: splitting, metrics, selection rules, LOBO."""

import numpy as np
import pandas as pd
import pytest

from hybrid_vpp.evaluation.blocked_validation import (
    SELECTION_RULES,
    SelectionWeights,
    candidate_metrics,
    contiguous_blocks,
    cvar,
    leave_one_block_out,
    metrics_table,
    selection_score,
)

DAYS = pd.date_range("2025-11-01", periods=92, freq="D")


def test_blocks_are_contiguous_ordered_and_cover_everything():
    blocks = contiguous_blocks(DAYS, 6)
    assert len(blocks) == 6
    rejoined = blocks[0]
    for b in blocks[1:]:
        assert b[0] > rejoined[-1]  # strictly later: no overlap, no shuffle
        rejoined = rejoined.append(b)
    assert rejoined.equals(DAYS)
    sizes = [len(b) for b in blocks]
    assert max(sizes) - min(sizes) <= 1


def test_blocks_reject_unsorted_duplicate_and_bad_counts():
    with pytest.raises(ValueError):
        contiguous_blocks(DAYS[::-1], 4)
    with pytest.raises(ValueError):
        contiguous_blocks(DAYS[:5].append(DAYS[:1]), 2)
    with pytest.raises(ValueError):
        contiguous_blocks(DAYS, 0)
    with pytest.raises(ValueError):
        contiguous_blocks(DAYS, len(DAYS) + 1)


def test_cvar_is_lower_tail_mean():
    assert cvar(np.arange(1, 11), alpha=0.2) == pytest.approx(1.5)  # mean of {1,2}
    assert cvar([5.0], alpha=0.1) == 5.0


def test_candidate_metrics_hand_computed():
    days = DAYS[:6]
    blocks = contiguous_blocks(days, 2)
    revenue = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], index=days)
    reference = pd.Series([20.0] * 6, index=days)
    m = candidate_metrics(revenue, reference, blocks, cvar_alpha=1 / 3)
    assert m["mean"] == pytest.approx(35.0)
    assert m["worst_fold"] == pytest.approx(20.0)  # first block mean
    assert m["cvar_regret"] == pytest.approx(-5.0)  # worst 2 regrets: -10, 0
    assert m["p_beat_reference"] == pytest.approx(4 / 6)
    assert m["upside_capture"] == pytest.approx((10 + 20 + 30 + 40) / 6)
    assert m["downside_exposure"] == pytest.approx(10 / 6)
    assert m["max_daily_loss_vs_reference"] == pytest.approx(10.0)


def test_candidate_metrics_requires_full_reference_coverage():
    days = DAYS[:4]
    with pytest.raises(ValueError):
        candidate_metrics(
            pd.Series(1.0, index=days),
            pd.Series(1.0, index=days[:2]),
            contiguous_blocks(days, 2),
        )


def test_selection_score_penalizes_dispersion_and_downside():
    base = {"mean": 100.0, "fold_std": 0.0, "cvar_regret": 0.0, "mean_regret": 0.0}
    w = SelectionWeights(fold_std=0.5, cvar_regret=0.5, negative_regret=0.25)
    assert selection_score(base, w) == 100.0
    assert selection_score({**base, "fold_std": 10.0}, w) == 95.0
    assert selection_score({**base, "cvar_regret": -20.0}, w) == 90.0
    assert selection_score({**base, "mean_regret": -8.0}, w) == 98.0
    # positive regret and positive tail are not rewarded twice
    assert selection_score({**base, "cvar_regret": 20.0, "mean_regret": 8.0}, w) == 100.0


def _toy_table() -> pd.DataFrame:
    days = DAYS[:30]
    blocks = contiguous_blocks(days, 3)
    rng = np.random.default_rng(7)
    reference = pd.Series(100.0 + rng.normal(0, 1, 30), index=days)
    candidates = pd.DataFrame(
        {
            # high mean, volatile: great two blocks, awful last block
            "volatile": np.r_[np.full(20, 140.0), np.full(10, 40.0)],
            # steady: slightly above reference everywhere
            "steady": np.full(30, 105.0),
            # bad: always below reference
            "bad": np.full(30, 80.0),
        },
        index=days,
    )
    return metrics_table(candidates, reference, blocks)


def test_selection_rules_pick_expected_candidates():
    table = _toy_table()
    assert SELECTION_RULES["highest_mean"](table) == "volatile"  # 106.7 mean
    assert SELECTION_RULES["highest_worst_fold"](table) == "steady"
    assert SELECTION_RULES["lowest_downside"](table) == "steady"
    assert SELECTION_RULES["risk_adjusted"](table) == "steady"
    assert SELECTION_RULES["pareto"](table) in ("volatile", "steady")  # never 'bad'


def test_lobo_never_sees_the_held_out_block():
    """A candidate that only shines in one block must not be chosen when that
    block is held out (fold leakage guard)."""
    days = DAYS[:30]
    blocks = contiguous_blocks(days, 3)
    reference = pd.Series(100.0, index=days)
    per_day = pd.DataFrame(
        {
            # spikes only inside block 2 — invisible to selection without leakage
            "leaky": np.r_[np.full(20, 90.0), np.full(10, 500.0)],
            "steady": np.full(30, 110.0),
        },
        index=days,
    )
    result = leave_one_block_out(per_day, reference, blocks)
    held2 = result[result.held_block == 2]
    assert (held2.chosen == "steady").all()
    assert (held2.oracle == "leaky").all()  # hindsight oracle does see it


def test_lobo_reports_reference_and_oracle_gap():
    days = DAYS[:20]
    blocks = contiguous_blocks(days, 2)
    reference = pd.Series(100.0, index=days)
    per_day = pd.DataFrame({"a": np.full(20, 110.0), "b": np.full(20, 105.0)}, index=days)
    result = leave_one_block_out(per_day, reference, blocks)
    assert (result.chosen == "a").all()
    assert (result.gap_to_oracle == 0.0).all()
    assert result.holdout_regret_mean.to_numpy() == pytest.approx(10.0)
    with pytest.raises(ValueError):
        leave_one_block_out(per_day, reference, [pd.DatetimeIndex(days)])
