"""Run the promoted ensemble deployment controller on one day.

The controller is the study's promoted design: the mean strategic action
of the five released SAC policies, clipped to a bounded residual around
the rule-equivalent action, with deterministic rule-based dispatch.

Model files are not stored in the repository. Download the five
checkpoints from the GitHub release (assets ``sac_hybrid_seed{0..4}.zip``)
into ``models/`` first, or point ``MODEL_DIR`` at your own training runs.
Requires the ``rl`` extra (stable-baselines3 + torch).

Run as-is:  uv run python examples/run_deployment_controller.py
"""

from pathlib import Path

import numpy as np

# --------------------------------------------------------------------- CONFIG

CONFIG_PATH = Path("configs/synthetic_market.yaml")
MODEL_DIR = Path("models")  # sac_hybrid_seed0.zip ... sac_hybrid_seed4.zip
RESIDUAL_BOUND = 0.1
DAY = "2025-05-14"

# ------------------------------------------------------------------------


def main(
    config_path: Path = CONFIG_PATH,
    model_dir: Path = MODEL_DIR,
    day: str = DAY,
    residual_bound: float = RESIDUAL_BOUND,
) -> float:
    checkpoints = sorted(model_dir.glob("sac_hybrid_seed*.zip"))
    if len(checkpoints) < 2:
        raise SystemExit(
            f"no ensemble checkpoints in {model_dir}/ — download the "
            "sac_hybrid_seed*.zip assets from the GitHub release "
            "(see docs/reproducibility.md) or train your own with "
            "configs/train_sac_hybrid.yaml"
        )

    from hybrid_vpp.config.models import load_config
    from hybrid_vpp.controllers import (
        EnsembleDeploymentController,
        PolicyEnsemble,
        SafetyGate,
        rule_equivalent_action,
    )
    from hybrid_vpp.envs.hybrid_vpp_env import HybridVppEnv
    from hybrid_vpp.training.algorithms import algo_class

    cfg = load_config(config_path)
    cfg.episode.action_mode = "strategic"
    cfg.episode.strategic_gain_max = 1.25
    cfg.episode.strategic_fixed_dispatch = True
    env = HybridVppEnv(cfg, split="train")

    models = [algo_class("sac").load(p) for p in checkpoints]
    rule = rule_equivalent_action(cfg.episode.strategic_gain_max)
    controller = EnsembleDeploymentController(
        ensemble=PolicyEnsemble(models, mode="mean"),
        gate=SafetyGate("bounded_residual", rule, max_residual=residual_bound),
        obs_low=np.full(env.observation_space.shape, -1e6),
        obs_high=np.full(env.observation_space.shape, 1e6),
        model_versions=tuple(p.name for p in checkpoints),
    )

    obs, info = env.reset(options={"day": day})
    done = False
    while not done:
        action = controller.act(obs, info["event_type"])
        obs, _, done, _, info = env.step(action)
    revenue = info["episode_metrics"]["total_net_revenue_eur"]
    fallbacks = sum(r["reason"] != "gated_rl" for r in controller.decision_log)
    print(f"delivery day {day}: net revenue {revenue:,.2f} EUR")
    print(f"decisions logged: {len(controller.decision_log)} (fallbacks: {fallbacks})")
    return revenue


if __name__ == "__main__":
    main()
