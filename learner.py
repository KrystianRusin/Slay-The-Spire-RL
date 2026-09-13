"""The learner: consumes rollouts from the broker and applies a PPO update for each."""

from contextlib import closing

import torch as th
from sb3_contrib.ppo_mask import MaskablePPO

from broker.config import load_broker_config
from broker.rollouts import consume_rollouts
from broker.topics import ensure_topics
from model.model_utils import update_model
from model.rollout_codec import decode_rollout
from slay_the_spire_env import SlayTheSpireEnv

SAVE_PATH = "maskable_ppo_slay_the_spire"


def train(model, rollouts, total_steps, save_path=SAVE_PATH):
    """Update the model on each encoded rollout as it arrives, until total_steps are consumed.

    Returns the number of steps consumed.
    """
    current_step = 0
    for encoded in rollouts:
        rollout = decode_rollout(encoded, model.observation_space, model.action_space, model.device)
        update_model(model, rollout, current_step, total_steps)
        model.save(save_path)
        current_step += len(rollout)
        if current_step >= total_steps:
            break
    return current_step


def main():
    config = load_broker_config()
    ensure_topics(config)

    def linear_clip_range(progress_remaining):
        return 0.3 * progress_remaining

    model = MaskablePPO(
        "MultiInputPolicy",
        SlayTheSpireEnv({}),
        ent_coef=0.03,
        gamma=0.97,
        learning_rate=0.0003,
        clip_range=linear_clip_range,
        verbose=1,
        device=th.device("cuda" if th.cuda.is_available() else "cpu")
    )

    with closing(consume_rollouts(config)) as rollouts:
        train(model, rollouts, total_steps=100000)


if __name__ == "__main__":
    main()
