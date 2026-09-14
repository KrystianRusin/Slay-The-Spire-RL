"""An actor: claims a game instance, plays it on the learner's newest published policy, and publishes each completed rollout to the broker."""

import argparse
import logging

from broker.config import load_broker_config
from broker.policy import PolicySubscription
from broker.rollouts import RolloutPublisher
from broker.topics import ensure_topics
from db.game_registry import HEARTBEAT_SECONDS, ClaimLost, GameRegistry, wait_for_game
from db.session import init_db
from environment.run_env import run_environment
from util.communication import MAX_RETRIES, Backoff, GameConnection
from util.heartbeat import Heartbeat

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", type=int, required=True, help="identifies this actor in logs, in the game registry and on the rollout topic")
    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES, help="reconnection attempts to its game before the actor gives up")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    init_db()
    config = load_broker_config()
    ensure_topics(config)
    publisher = RolloutPublisher(config, args.env_id)
    policy_source = PolicySubscription(config)
    registry = GameRegistry()
    actor_id = str(args.env_id)
    try:
        game_id = wait_for_game(registry, actor_id)
        logger.info("Actor %s claimed game %s", actor_id, game_id)
        connection = GameConnection(
            lambda: registry.address(game_id, actor_id),
            Backoff(max_retries=args.max_retries),
        )

        def renew_claim():
            try:
                registry.renew(game_id, actor_id)
            except ClaimLost as error:
                connection.abandon(error)
                raise

        try:
            with Heartbeat(HEARTBEAT_SECONDS, renew_claim, "game claim"):
                run_environment(args.env_id, connection, publisher, policy_source)
        finally:
            connection.close()
            registry.release(game_id, actor_id)
    finally:
        policy_source.close()
        publisher.close()


if __name__ == "__main__":
    main()
