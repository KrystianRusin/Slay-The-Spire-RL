"""The one place the policy network and its PPO settings are defined.

The learner builds its model from a TrainingConfig and sends that config to
actors with the weights (see rpc/learner.proto), so actors build the identical
network and load the weights into it. Change a setting here and every actor
picks it up the next time it fetches weights.
"""
from dataclasses import dataclass

from sb3_contrib.ppo_mask import MaskablePPO

from sts2.env import Sts2Env
from sts2.features import Sts2Extractor


@dataclass(frozen=True)
class TrainingConfig:
    # Used by actors: how long a rollout is, and how advantages/returns are computed.
    n_steps: int = 2048
    # A run is several hundred decisions, and a fight's payoff (the floor
    # reward) lands well after the cards that won it, so look further ahead.
    gamma: float = 0.99
    gae_lambda: float = 0.95
    # Actors multiply every reward by this, so returns come out around unit size
    # and the value loss doesn't swamp the policy's share of the gradient. The
    # raw rewards (sts2/env.py) are already about unit size per floor.
    reward_scale: float = 0.2
    # Used by the learner's update.
    learning_rate: float = 2.5e-4
    ent_coef: float = 0.01
    clip_range: float = 0.2
    n_epochs: int = 4
    batch_size: int = 512
    # Hidden layer sizes after the feature extractor (sts2/features.py). The
    # weights only fit a network with the same layers.
    policy_layers: tuple = (256, 256)
    value_layers: tuple = (256, 256)


def make_model(config, env=None, device="cpu"):
    """Build a MaskablePPO with fresh random weights from config.

    env only supplies the observation and action spaces; it is never stepped here.
    """
    return MaskablePPO(
        "MultiInputPolicy",
        env if env is not None else Sts2Env(),
        n_steps=config.n_steps,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        learning_rate=config.learning_rate,
        ent_coef=config.ent_coef,
        clip_range=config.clip_range,
        n_epochs=config.n_epochs,
        batch_size=config.batch_size,
        policy_kwargs={"net_arch": {"pi": list(config.policy_layers),
                                    "vf": list(config.value_layers)},
                       "features_extractor_class": Sts2Extractor},
        device=device,
    )
