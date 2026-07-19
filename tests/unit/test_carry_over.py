"""Battery SoC carry-over across episodes: default off, config flag,
explicit override, and boundary valuation against the actual start energy."""

from pathlib import Path

import numpy as np
import pytest

from hybrid_vpp.config.models import load_config
from hybrid_vpp.envs.hybrid_vpp_env import HybridVppEnv

CONFIG = Path(__file__).parents[2] / "configs" / "synthetic_market.yaml"


def make_env(carry_over: bool) -> HybridVppEnv:
    cfg = load_config(CONFIG)
    cfg.episode.action_mode = "residual_hourly"
    cfg.episode.carry_over_soc = carry_over
    return HybridVppEnv(cfg, split="train")


def run_day(env: HybridVppEnv, day: str, initial_soc: float | None = None) -> dict:
    options: dict = {"day": day}
    if initial_soc is not None:
        options["initial_soc"] = initial_soc
    env.reset(seed=0, options=options)
    done = False
    while not done:
        _, _, done, _, info = env.step(np.zeros(env.action_space.shape, np.float32))
    return info["episode_metrics"]


def test_default_resets_to_soc_initial():
    env = make_env(carry_over=False)
    cap = env.cfg.site.battery.energy_capacity_mwh
    run_day(env, "2025-05-12")
    run_day(env, "2025-05-13")
    assert env.sim.episode_start_energy_mwh == pytest.approx(0.5 * cap)


def test_carry_over_chains_final_soc():
    env = make_env(carry_over=True)
    m1 = run_day(env, "2025-05-12")
    end_energy = env.sim.battery.energy_mwh
    assert m1["final_soc"] != pytest.approx(0.5)  # day must actually move the battery
    run_day(env, "2025-05-13")
    assert env.sim.episode_start_energy_mwh == pytest.approx(end_energy, abs=1e-9)


def test_explicit_initial_soc_wins():
    env = make_env(carry_over=True)
    cap = env.cfg.site.battery.energy_capacity_mwh
    run_day(env, "2025-05-12")
    run_day(env, "2025-05-13", initial_soc=0.8)
    assert env.sim.episode_start_energy_mwh == pytest.approx(0.8 * cap)


def test_initial_soc_clamped_to_window():
    env = make_env(carry_over=False)
    bat = env.cfg.site.battery
    env.reset(seed=0, options={"day": "2025-05-12", "initial_soc": 1.0})
    assert env.sim.battery.soc == pytest.approx(bat.soc_max)


def test_initial_soc_range_randomizes_start():
    cfg = load_config(CONFIG)
    cfg.episode.action_mode = "residual_hourly"
    cfg.episode.initial_soc_range = (0.05, 0.95)
    env = HybridVppEnv(cfg, split="train")
    bat = env.cfg.site.battery
    starts = []
    for _ in range(5):
        env.reset(seed=None, options={"day": "2025-05-12"})
        starts.append(env.sim.episode_start_energy_mwh / bat.energy_capacity_mwh)
    assert all(bat.soc_min <= s <= bat.soc_max for s in starts)
    assert np.std(starts) > 0.01  # actually randomized, not stuck at 0.5


def test_terminal_value_uses_actual_start_energy():
    env = make_env(carry_over=False)
    cap = env.cfg.site.battery.energy_capacity_mwh
    m = run_day(env, "2025-05-12", initial_soc=0.8)
    expected = (env.sim.battery.energy_mwh - 0.8 * cap) * env.sim.episode_mean_daa_price_eur_mwh()
    assert m["terminal_energy_value_eur"] == pytest.approx(expected, abs=1e-9)
    assert m["total_net_revenue_terminal_adjusted_eur"] == pytest.approx(
        m["total_net_revenue_eur"] + expected, abs=1e-9
    )
