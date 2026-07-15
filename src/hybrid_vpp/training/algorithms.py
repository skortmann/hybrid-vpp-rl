"""Common algorithm adapter: identical envs, logging, and evaluation for all
candidates; per-algorithm defaults tuned to this environment's scale.

All algorithms consume the same Gymnasium environment (any action layout) —
the adapter only maps names to classes and supplies default hyperparameters,
so comparisons differ in nothing but the algorithm itself.
"""

from __future__ import annotations

ALGORITHMS = ("ppo", "sac", "tqc", "td3", "recurrent_ppo")


def algo_class(name: str):
    if name == "ppo":
        from stable_baselines3 import PPO

        return PPO
    if name == "sac":
        from stable_baselines3 import SAC

        return SAC
    if name == "td3":
        from stable_baselines3 import TD3

        return TD3
    if name == "tqc":
        from sb3_contrib import TQC

        return TQC
    if name == "recurrent_ppo":
        from sb3_contrib import RecurrentPPO

        return RecurrentPPO
    raise ValueError(f"unknown algorithm {name!r} (choose from {ALGORITHMS})")


def default_kwargs(name: str) -> dict:
    """Environment-scale-aware defaults (episodes of ~60-132 steps, kEUR rewards)."""
    if name == "ppo":
        return dict(
            n_steps=256,
            batch_size=1024,
            gamma=0.995,
            gae_lambda=0.95,
            learning_rate=3e-4,
            ent_coef=0.0,
            clip_range=0.2,
            n_epochs=10,
            target_kl=0.03,
            policy_kwargs={"net_arch": [256, 256], "log_std_init": -1.0},
        )
    if name == "recurrent_ppo":
        return dict(
            n_steps=256,
            batch_size=1024,
            gamma=0.995,
            gae_lambda=0.95,
            learning_rate=3e-4,
            ent_coef=0.0,
            clip_range=0.2,
            n_epochs=10,
            target_kl=0.03,
            policy_kwargs={"lstm_hidden_size": 128, "net_arch": [256], "log_std_init": -1.0},
        )
    if name in ("sac", "tqc"):
        kwargs = dict(
            learning_rate=3e-4,
            buffer_size=300_000,
            learning_starts=5_000,
            batch_size=256,
            tau=0.005,
            gamma=0.995,
            train_freq=1,
            gradient_steps=1,
            policy_kwargs={"net_arch": [256, 256]},
        )
        if name == "tqc":
            kwargs["policy_kwargs"] = {
                "net_arch": [256, 256],
                "n_critics": 2,
                "n_quantiles": 25,
            }
            kwargs["top_quantiles_to_drop_per_net"] = 2
        return kwargs
    if name == "td3":
        return dict(
            learning_rate=1e-3,
            buffer_size=300_000,
            learning_starts=5_000,
            batch_size=256,
            tau=0.005,
            gamma=0.995,
            train_freq=1,
            gradient_steps=1,
            policy_delay=2,
            policy_kwargs={"net_arch": [256, 256]},
        )
    raise ValueError(f"unknown algorithm {name!r}")


def policy_name(name: str) -> str:
    return "MlpLstmPolicy" if name == "recurrent_ppo" else "MlpPolicy"
