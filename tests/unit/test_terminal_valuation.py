"""Terminal-adjusted revenue metric: identity, plumbing, and consistency
with the training reward's boundary valuation (synthetic env)."""

from pathlib import Path

import numpy as np
import pytest

from hybrid_vpp.config.models import load_config
from hybrid_vpp.envs.hybrid_vpp_env import HybridVppEnv

CONFIG = Path(__file__).parents[2] / "configs" / "synthetic_market.yaml"


@pytest.fixture(scope="module")
def finished_env() -> HybridVppEnv:
    cfg = load_config(CONFIG)
    cfg.episode.action_mode = "strategic"
    env = HybridVppEnv(cfg, split="train")
    a = np.array([2 / 1.2 - 1, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    env.reset(seed=0, options={"day": "2025-05-14"})
    done = False
    while not done:
        _, _, done, _, info = env.step(a)
    env._last_info = info
    return env


def test_adjusted_equals_raw_plus_terminal_value(finished_env):
    m = finished_env._last_info["episode_metrics"]
    assert m["total_net_revenue_terminal_adjusted_eur"] == pytest.approx(
        m["total_net_revenue_eur"] + m["terminal_energy_value_eur"], abs=1e-9
    )


def test_terminal_value_matches_battery_state(finished_env):
    env = finished_env
    m = env._last_info["episode_metrics"]
    bat = env.cfg.site.battery
    e0 = bat.soc_initial * bat.energy_capacity_mwh
    expected = (env.sim.battery.energy_mwh - e0) * env.sim.episode_mean_daa_price_eur_mwh()
    assert m["terminal_energy_value_eur"] == pytest.approx(expected, abs=1e-9)


def test_terminal_value_consistent_with_training_reward(finished_env):
    """The metric applies the same valuation rule as the terminal reward.

    Small differences are allowed: the reward's price window uses local-day
    bounds while the metric averages over the episode's delivery products.
    """
    env = finished_env
    m = env._last_info["episode_metrics"]
    reward_value = env._terminal_soc_value()
    if abs(reward_value) < 1.0:  # both ~zero when the day ends near initial SoC
        assert abs(m["terminal_energy_value_eur"]) < 1.0
    else:
        assert m["terminal_energy_value_eur"] == pytest.approx(reward_value, rel=0.1)
