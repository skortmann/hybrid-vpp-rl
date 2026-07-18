"""Strategic-residual action mode (act-v6): translator semantics and env integration."""

from pathlib import Path

import numpy as np
import pytest

from hybrid_vpp.config.models import load_config
from hybrid_vpp.envs.hybrid_vpp_env import HybridVppEnv

CONFIG = Path(__file__).parents[2] / "configs" / "synthetic_market.yaml"

#: strategic head that reproduces the rule-based controller (see act-v5 tests)
MIDRANGE = [2 / 1.2 - 1, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
#: an arbitrary interior strategic head (used for the v5 ≡ v6 equivalence check)
INTERIOR = [0.2, -0.3, 0.5, 0.1, -0.2, 0.4, 0.1]


def make_env(mode: str) -> HybridVppEnv:
    cfg = load_config(CONFIG)
    cfg.episode.action_mode = mode
    return HybridVppEnv(cfg, split="train")


@pytest.fixture(scope="module")
def env() -> HybridVppEnv:
    return make_env("strategic_residual")


@pytest.fixture(scope="module")
def val_env() -> HybridVppEnv:
    """Post-MTU-switch split: DAA products are quarter-hourly."""
    cfg = load_config(CONFIG)
    cfg.episode.action_mode = "strategic_residual"
    return HybridVppEnv(cfg, split="val")


def action(env, strategic=MIDRANGE, anchors=(), tilts=()) -> np.ndarray:
    """57-dim act-v6 vector: 7 strategic + 25 anchors + 25 tilts (1-day episodes)."""
    a = np.zeros(env.action_space.shape, dtype=np.float32)
    a[:7] = strategic
    for hour, value in anchors:
        a[7 + hour] = value
    for hour, value in tilts:
        a[7 + 25 + hour] = value
    return a


def test_action_and_observation_sizes(env):
    assert env.action_space.shape == (57,)  # 7 strategic + 25 anchors + 25 tilts
    # observations are quarter-hourly in this mode
    assert env.observation_space.shape[0] == 23 + 5 * 100


def test_event_masks(env):
    obs, info = env.reset(seed=0, options={"day": "2025-05-12"})
    assert info["event_type"] == "DAA_GATE_CLOSURE"
    mask = info["action_mask"]
    assert mask[:7].tolist() == [1, 1, 0, 0, 0, 0, 0]
    # a regular 24-hour day: anchors/tilts of hours 0..23 active, hour 24 dead
    assert mask[7 : 7 + 24].tolist() == [1] * 24
    assert mask[7 + 24] == 0
    assert mask[32 : 32 + 24].tolist() == [1] * 24
    assert mask[32 + 24] == 0


def test_zero_residual_reproduces_strategic(env):
    """act-v6 with zero residual coefficients is exactly act-v5."""
    day = "2025-05-14"
    obs, _ = env.reset(seed=0, options={"day": day})
    done = False
    a6 = action(env, strategic=INTERIOR)
    while not done:
        obs, r, done, _, info = env.step(a6)
    v6_total = env.sim.ledger.total()

    env5 = make_env("strategic")
    env5.reset(seed=0, options={"day": day})
    a5 = np.array(INTERIOR, dtype=np.float32)
    done = False
    while not done:
        obs, r, done, _, info = env5.step(a5)
    assert v6_total == pytest.approx(env5.sim.ledger.total(), rel=1e-9)


def test_midrange_zero_residual_reproduces_rule_based(env):
    day = "2025-05-14"
    obs, _ = env.reset(seed=0, options={"day": day})
    done = False
    while not done:
        obs, r, done, _, info = env.step(action(env))
    v6_total = env.sim.ledger.total()

    from hybrid_vpp.controllers.base import run_episode
    from hybrid_vpp.controllers.rule_based import RuleBasedController

    controller = RuleBasedController(
        env.cfg,
        env.obs_builder.renewable_forecaster,
        env.obs_builder.price_forecaster,
    )
    run_episode(env.sim, controller, day)
    assert v6_total == pytest.approx(env.sim.ledger.total(), rel=1e-6)


def test_anchor_shifts_hour_uniformly(val_env):
    """On a quarter-hourly DAA day, one anchor moves all four quarter-hours alike."""
    env = val_env
    env.reset(seed=0, options={"day": "2025-10-15"})
    # coverage 0 / arbitrage 0: no strategic DAA orders, the anchor acts alone
    a = action(env, strategic=[-1, -1, 0, 0, 0, 0, 0], anchors=[(10, 1.0)])
    env.step(a)  # DAA gate
    scale = env.cfg.episode.residual_scale_mw
    w0 = env._window_start
    qhs = [w0 + np.timedelta64(10 * 60 + q * 15, "m") for q in range(4)]
    values = [env.sim.book.net_position_mw_at(t) for t in qhs]
    assert np.allclose(values, scale, atol=1e-6)
    # neighbouring hour untouched
    assert env.sim.book.net_position_mw_at(w0 + np.timedelta64(11, "h")) == pytest.approx(0.0)


def test_tilt_is_zero_mean_on_quarter_hourly_day(val_env):
    env = val_env
    env.reset(seed=0, options={"day": "2025-10-15"})
    a = action(env, strategic=[-1, -1, 0, 0, 0, 0, 0], tilts=[(10, 1.0)])
    env.step(a)  # DAA gate
    scale = env.cfg.episode.intra_hour_residual_scale_mw
    w0 = env._window_start
    qhs = [w0 + np.timedelta64(10 * 60 + q * 15, "m") for q in range(4)]
    values = [env.sim.book.net_position_mw_at(t) for t in qhs]
    expected = [scale * b for b in (-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0)]
    assert np.allclose(values, expected, atol=1e-6)
    assert sum(values) == pytest.approx(0.0, abs=1e-6)  # net hour volume unchanged


def test_tilt_has_no_effect_on_hourly_products(env):
    """Pre-switch DAA products span the whole hour: a zero-mean tilt cancels."""
    env.reset(seed=0, options={"day": "2025-05-15"})
    a = action(env, strategic=[-1, -1, 0, 0, 0, 0, 0], tilts=[(h, 1.0) for h in range(24)])
    env.step(a)  # DAA gate (hourly products)
    assert env.sim.book.turnover_mwh("daa") == pytest.approx(0.0)


def test_full_episode_random(env):
    obs, _ = env.reset(seed=3, options={"day": "2025-05-20"})
    done = False
    while not done:
        obs, r, done, _, info = env.step(env.action_space.sample() * 0.5)
        assert np.isfinite(r)
    m = info["episode_metrics"]
    assert m["daa_volume_mwh"] >= 0
    assert env.sim.ledger.total() == pytest.approx(sum(env.sim.ledger.by_component().values()))


def test_capped_orders_are_surfaced_in_info():
    """Execution volume caps are reported via info, not silently absorbed."""
    cfg = load_config(CONFIG)
    cfg.episode.action_mode = "residual_hourly"
    cfg.markets.daa.max_volume_mw = 3.0  # far below typical rule-based orders
    env = HybridVppEnv(cfg, split="train")
    env.reset(seed=0, options={"day": "2025-05-12"})
    obs, r, done, _, info = env.step(np.zeros(env.action_space.shape, np.float32))  # DAA
    assert info["orders_submitted"] > 0
    assert info["orders_capped"] > 0
    assert info["capped_mw"] > 0.0
