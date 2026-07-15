"""End-to-end pipeline on the synthetic database — no real DB required.

Verifies the drop-in property: identical env/controller code paths run on
synthetic data with the same observation/action structure and accounting.
"""

from pathlib import Path

import numpy as np
import pytest

from hybrid_vpp.config.models import load_config
from hybrid_vpp.controllers.base import run_episode
from hybrid_vpp.controllers.rule_based import RuleBasedController
from hybrid_vpp.envs.hybrid_vpp_env import HybridVppEnv
from hybrid_vpp.evaluation.metrics import episode_metrics
from hybrid_vpp.evaluation.run_baselines import build_stack
from tests.conftest import REAL_DB_AVAILABLE

CONFIG = Path(__file__).parents[2] / "configs" / "synthetic_market.yaml"
REAL_CONFIG = Path(__file__).parents[2] / "configs" / "default.yaml"


@pytest.fixture(scope="module")
def stack():
    cfg = load_config(CONFIG)
    return cfg, *build_stack(cfg)


def test_provenance_is_synthetic(stack):
    cfg, store, *_ = stack
    assert store.provenance == "synthetic"
    assert store.source.metadata["synthetic"] == "true"


def test_rule_based_episode_on_synthetic(stack):
    cfg, store, profiles, sim, renewable_fc, price_fc = stack
    controller = RuleBasedController(cfg, renewable_fc, price_fc)
    run_episode(sim, controller, "2025-05-10")
    metrics = episode_metrics(sim)
    assert metrics["daa_volume_mwh"] > 0
    assert sim.ledger.total() == pytest.approx(sum(sim.ledger.by_component().values()))
    for record in sim.dispatch_records.values():
        assert record.dispatch.grid_power_mw <= cfg.site.grid.export_limit_mw + 1e-6


def test_env_runs_unchanged_on_synthetic(stack):
    cfg, store, profiles, *_ = stack
    env = HybridVppEnv(cfg, split="train", store=store, profiles=profiles)
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    total, done = 0.0, False
    while not done:
        obs, reward, done, _, info = env.step(env.action_space.sample() * 0.1)
        assert np.isfinite(reward)
        total += reward
    assert "episode_metrics" in info


@pytest.mark.skipif(not REAL_DB_AVAILABLE, reason="real database not available")
def test_observation_structure_matches_real(stack):
    """Same env code on real vs. synthetic data: identical spaces and masks."""
    cfg_syn, store_syn, profiles_syn, *_ = stack
    env_syn = HybridVppEnv(cfg_syn, split="train", store=store_syn, profiles=profiles_syn)
    cfg_real = load_config(REAL_CONFIG)
    env_real = HybridVppEnv(cfg_real, split="train")

    assert env_syn.observation_space.shape == env_real.observation_space.shape
    assert env_syn.action_space.shape == env_real.action_space.shape

    obs_s, info_s = env_syn.reset(seed=1)
    obs_r, info_r = env_real.reset(seed=1)
    assert info_s["event_type"] == info_r["event_type"] == "DAA_GATE_CLOSURE"
    assert obs_s.dtype == obs_r.dtype
    assert info_s["action_mask"].shape == info_r["action_mask"].shape
