"""The learner: consumes rollouts from the broker and applies a PPO update for each."""

import argparse
import logging
import os
import time
from contextlib import closing

import torch as th
from confluent_kafka import KafkaException
from dotenv import load_dotenv
from sb3_contrib.ppo_mask import MaskablePPO

from broker.config import load_broker_config
from broker.policy import PolicyPublisher
from broker.rollouts import ConsumerLagProbe, RolloutConsumer
from broker.topics import ensure_topics
from model.checkpoint import TrainingProgress, load_checkpoint, save_checkpoint
from model.model_utils import update_model
from model.policy_codec import encode_policy
from model.rollout_codec import SCHEMA_VERSION, UnsupportedSchemaVersion, decode_rollout
from observability.logs import configure_logging
from observability.metrics import LearnerMetrics
from slay_the_spire_env import SlayTheSpireEnv
from util.heartbeat import Heartbeat

SAVE_PATH = "maskable_ppo_slay_the_spire"
# Each rollout waiting is one more update the policy moves on before that rollout is used; see docs/adr/0003.
LAG_WARNING_ROLLOUTS = 5
LAG_CHECK_SECONDS = 10
METRICS_PORT_VAR = "LEARNER_METRICS_PORT"
DEFAULT_METRICS_PORT = 8000

logger = logging.getLogger(__name__)


def train(model, deliveries, total_steps, progress=None, save_path=SAVE_PATH, publisher=None, metrics=None):
    """Apply a PPO update for each delivered rollout until progress reaches total_steps.

    With a publisher, the current policy is published before training starts,
    and each update is published as the next policy version once it is saved.
    A delivery is committed only once its update is saved and published, so a
    rollout interrupted mid-update is delivered again. Rollouts the checkpoint
    already includes, and rollouts that cannot be decoded, are committed
    without being trained on. A rollout from a newer schema raises
    UnsupportedSchemaVersion uncommitted, since this learner is out of date.

    Returns the number of steps trained, counting those already in progress.
    """
    if progress is None:
        progress = TrainingProgress()
    if metrics is None:
        metrics = LearnerMetrics()
    metrics.record_progress(progress)
    _publish(model, progress, publisher)
    if progress.steps >= total_steps:
        return progress.steps

    for delivery in deliveries:
        try:
            rollout = decode_rollout(delivery.value, model.observation_space, model.action_space, model.device)
        except ValueError as error:
            if isinstance(error, UnsupportedSchemaVersion) and error.version > SCHEMA_VERSION:
                raise
            logger.error("Skipping a rollout that cannot be decoded: %s", error)
            metrics.skipped("undecodable")
        else:
            if progress.has_applied(rollout.rollout_id):
                logger.warning("Skipping rollout %s, which was already applied before a restart", rollout.rollout_id)
                metrics.skipped("already_applied")
            else:
                metrics.training_on(delivery.published_at)
                update_model(model, rollout, progress.steps, total_steps)
                progress.record(rollout.rollout_id, len(rollout))
                save_checkpoint(model, progress, save_path)
                _publish(model, progress, publisher)
                metrics.updated(progress)
        delivery.commit()
        if progress.steps >= total_steps:
            break
    return progress.steps


def check_lag(probe, metrics):
    """Measure the learner group's consumer lag, export it, and warn while it is at or above LAG_WARNING_ROLLOUTS."""
    try:
        lag = probe.measure()
    except KafkaException as error:
        logger.warning("Could not measure consumer lag: %s", error)
        metrics.consumer_lag(None)
        return
    metrics.consumer_lag(lag)
    total = sum(lag.values())
    if total >= LAG_WARNING_ROLLOUTS:
        logger.warning("Learner is %d rollouts behind the actors; the rollouts it trains on are getting stale", total)


def slowed(deliveries, seconds):
    """The deliveries, each handed over only after a pause of seconds."""
    for delivery in deliveries:
        time.sleep(seconds)
        yield delivery


def _publish(model, progress, publisher):
    if publisher is not None:
        publisher.publish(encode_policy(model.policy, progress.policy_version))
        logger.info("Published policy version %d", progress.policy_version)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update-delay", type=float, default=0.0, metavar="SECONDS",
                        help="pause before each update, to slow the learner down and watch consumer lag build")
    args = parser.parse_args()

    load_dotenv()
    configure_logging(service="learner")
    metrics = LearnerMetrics()
    metrics.serve(int(os.environ.get(METRICS_PORT_VAR, DEFAULT_METRICS_PORT)))
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
            verbose=0,
            device=device
        )
        progress = TrainingProgress()
    else:
        model, progress = checkpoint
        logger.info("Resuming from checkpoint at step %d", progress.steps)

    with (
        closing(PolicyPublisher(config)) as publisher,
        closing(RolloutConsumer(config)) as consumer,
        closing(ConsumerLagProbe(config)) as probe,
        Heartbeat(LAG_CHECK_SECONDS, lambda: check_lag(probe, metrics), "consumer lag"),
    ):
        deliveries = slowed(consumer, args.update_delay) if args.update_delay else consumer
        train(model, deliveries, total_steps=100000, progress=progress, publisher=publisher, metrics=metrics)


if __name__ == "__main__":
    main()
