"""Action-layout variants: translation semantics per mode (synthetic env)."""

from pathlib import Path

import numpy as np
import pytest

from hybrid_vpp.config.models import load_config
from hybrid_vpp.envs.hybrid_vpp_env import HybridVppEnv

CONFIG = Path(__file__).parents[2] / "configs" / "synthetic_market.yaml"


def make_env(mode: str) -> HybridVppEnv:
    cfg = load_config(CONFIG)
    cfg.episode.action_mode = mode
    return HybridVppEnv(cfg, split="train")


@pytest.fixture(scope="module")
def envs():
    return {
        mode: make_env(mode)
        for mode in ("direct", "target_position", "hourly_target", "residual_hourly")
    }


def test_action_sizes(envs):
    assert envs["direct"].action_space.shape == (103,)
    assert envs["target_position"].action_space.shape == (103,)
    assert envs["hourly_target"].action_space.shape == (28,)  # 25 hours + 3 dispatch
    assert envs["residual_hourly"].action_space.shape == (28,)


def test_observation_sizes_scale_with_slots(envs):
    assert envs["direct"].observation_space.shape[0] == 23 + 5 * 100
    assert envs["hourly_target"].observation_space.shape[0] == 23 + 5 * 25


def test_target_position_submits_delta_and_is_idempotent(envs):
    env = envs["target_position"]
    env.reset(seed=0, options={"day": "2025-05-12"})
    # DAA gate: request a flat cumulative target of 0.2 * max_volume = 30 MW
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    action[: env.layout.n_slots] = 0.2
    env.step(action)
    max_volume = env.cfg.markets.daa.max_volume_mw
    positions = [env.sim.book.net_position_mw_at(t) for t in env.obs_builder._slot_times[:96]]
    assert np.allclose(positions, 0.2 * max_volume, atol=1e-6)

    # the next auction gate with the same target must trade ~nothing extra
    trades_before = len(env.sim.book.trades)
    env.step(action)  # IDA1 gate
    new_trades = env.sim.book.trades[trades_before:]
    assert sum(t.volume_mw for t in new_trades) < 1e-6  # target already met


def test_hourly_target_broadcasts(envs):
    env = envs["hourly_target"]
    env.reset(seed=0, options={"day": "2025-05-12"})
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    action[: env.layout.n_slots] = 0.1
    env.step(action)  # DAA gate
    max_volume = env.cfg.markets.daa.max_volume_mw
    # all four quarter-hours of an hour share the anchor's target position
    for hour_offset in (7, 12, 20):
        starts = [
            env._window_start + np.timedelta64(hour_offset * 60 + q * 15, "m") for q in range(4)
        ]
        values = [env.sim.book.net_position_mw_at(t) for t in starts]
        assert np.allclose(values, 0.1 * max_volume, atol=1e-6)


def test_residual_zero_action_reproduces_rule_based(envs):
    env = envs["residual_hourly"]
    obs, _ = env.reset(seed=0, options={"day": "2025-05-12"})
    done = False
    while not done:
        obs, r, done, _, info = env.step(np.zeros(env.action_space.shape, np.float32))
    residual_total = env.sim.ledger.total()

    # rule-based controller on the same day through the plain simulator
    from hybrid_vpp.controllers.base import run_episode
    from hybrid_vpp.controllers.rule_based import RuleBasedController

    controller = RuleBasedController(
        env.cfg,
        env.obs_builder.renewable_forecaster,
        env.obs_builder.price_forecaster,
    )
    run_episode(env.sim, controller, "2025-05-12")
    rb_total = env.sim.ledger.total()
    assert residual_total == pytest.approx(rb_total, rel=1e-9)


def test_residual_nonzero_changes_orders(envs):
    env = envs["residual_hourly"]
    env.reset(seed=0, options={"day": "2025-05-13"})
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    action[: env.layout.n_slots] = 1.0  # +residual_scale on every hour
    env.step(action)
    scale = env.cfg.episode.residual_scale_mw
    # position should exceed a zero-action baseline by roughly the scale
    env2 = make_env("residual_hourly")
    env2.reset(seed=0, options={"day": "2025-05-13"})
    env2.step(np.zeros(env2.action_space.shape, np.float32))
    t0 = env._window_start
    delta = env.sim.book.net_position_mw_at(t0) - env2.sim.book.net_position_mw_at(t0)
    assert delta == pytest.approx(scale, abs=1.0)


def test_all_modes_run_full_episode(envs):
    for mode, env in envs.items():
        obs, _ = env.reset(seed=1, options={"day": "2025-05-20"})
        done = False
        while not done:
            obs, r, done, _, info = env.step(env.action_space.sample() * 0.1)
            assert np.isfinite(r), mode
        assert "episode_metrics" in info, mode
