"""The learner: consumes rollouts from the broker and applies a PPO update for each."""

import logging
from contextlib import closing

import torch as th
from sb3_contrib.ppo_mask import MaskablePPO

from broker.config import load_broker_config
from broker.rollouts import RolloutConsumer
from broker.topics import ensure_topics
from model.checkpoint import TrainingProgress, load_checkpoint, save_checkpoint
from model.model_utils import update_model
from model.rollout_codec import SCHEMA_VERSION, UnsupportedSchemaVersion, decode_rollout
from slay_the_spire_env import SlayTheSpireEnv

SAVE_PATH = "maskable_ppo_slay_the_spire"

logger = logging.getLogger(__name__)


def train(model, deliveries, total_steps, progress=None, save_path=SAVE_PATH):
    """Apply a PPO update for each delivered rollout until progress reaches total_steps.

    A delivery is committed only once its update is in a saved checkpoint, so a
    rollout interrupted mid-update is delivered again. Rollouts the checkpoint
    already includes, and rollouts that cannot be decoded, are committed
    without being trained on. A rollout from a newer schema raises
    UnsupportedSchemaVersion uncommitted, since this learner is out of date.

    Returns the number of steps trained, counting those already in progress.
    """
    if progress is None:
        progress = TrainingProgress()
    if progress.steps >= total_steps:
        return progress.steps

    for delivery in deliveries:
        try:
            rollout = decode_rollout(delivery.value, model.observation_space, model.action_space, model.device)
        except ValueError as error:
            if isinstance(error, UnsupportedSchemaVersion) and error.version > SCHEMA_VERSION:
                raise
            logger.error("Skipping a rollout that cannot be decoded: %s", error)
        else:
            if progress.has_applied(rollout.rollout_id):
                logger.warning("Skipping rollout %s, which was already applied before a restart", rollout.rollout_id)
            else:
                update_model(model, rollout, progress.steps, total_steps)
                progress.record(rollout.rollout_id, len(rollout))
                save_checkpoint(model, progress, save_path)
        delivery.commit()
        if progress.steps >= total_steps:
            break
    return progress.steps


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = load_broker_config()
    ensure_topics(config)
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    def linear_clip_range(progress_remaining):
        return 0.3 * progress_remaining

    env = SlayTheSpireEnv({})
    checkpoint = load_checkpoint(SAVE_PATH, env, device)
    if checkpoint is None:
        model = MaskablePPO(
            "MultiInputPolicy",
            env,
            ent_coef=0.03,
            gamma=0.97,
            learning_rate=0.0003,
            clip_range=linear_clip_range,
            verbose=1,
            device=device
        )
        progress = TrainingProgress()
    else:
        model, progress = checkpoint
        logger.info("Resuming from checkpoint at step %d", progress.steps)

    with closing(RolloutConsumer(config)) as deliveries:
        train(model, deliveries, total_steps=100000, progress=progress)


if __name__ == "__main__":
    main()
