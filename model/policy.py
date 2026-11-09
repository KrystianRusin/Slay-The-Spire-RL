"""The one place the policy network and its PPO settings are defined.

The learner builds its model from a TrainingConfig and sends that config to
actors with the weights (see rpc/learner.proto), so actors build the identical
network and load the weights into it. Change a setting here and every actor
picks it up the next time it fetches weights.
"""
from dataclasses import dataclass

from sb3_contrib.ppo_mask import MaskablePPO

from slay_the_spire_env import SlayTheSpireEnv


@dataclass(frozen=True)
class TrainingConfig:
    # Used by actors: how long a rollout is, and how advantages/returns are computed.
    n_steps: int = 2048
    gamma: float = 0.97
    gae_lambda: float = 0.95
    # Used by the learner's update.
    learning_rate: float = 3e-4
    ent_coef: float = 0.03
    clip_range: float = 0.3
    n_epochs: int = 10
    batch_size: int = 64
    # Hidden layer sizes. The weights only fit a network with the same layers.
    policy_layers: tuple = (64, 64)
    value_layers: tuple = (64, 64)


def make_model(config, env=None, device="cpu"):
    """Build a MaskablePPO with fresh random weights from config.

    env only supplies the observation and action spaces; it is never stepped here.
    """
    return MaskablePPO(
        "MultiInputPolicy",
        env if env is not None else SlayTheSpireEnv({}),
        n_steps=config.n_steps,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        learning_rate=config.learning_rate,
        ent_coef=config.ent_coef,
        clip_range=config.clip_range,
        n_epochs=config.n_epochs,
        batch_size=config.batch_size,
        policy_kwargs={"net_arch": {"pi": list(config.policy_layers),
                                    "vf": list(config.value_layers)}},
        device=device,
    )
