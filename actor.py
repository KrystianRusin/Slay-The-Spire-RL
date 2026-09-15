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
from observability.logs import configure_logging
from observability.metrics import ActorMetrics
from util.communication import MAX_RETRIES, Backoff, GameConnection
from util.heartbeat import Heartbeat

METRICS_BASE_PORT = 8100

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", type=int, required=True, help="identifies this actor in logs, in the game registry and on the rollout topic")
    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES, help="reconnection attempts to its game before the actor gives up")
    parser.add_argument("--metrics-port", type=int, help=f"port to serve Prometheus metrics on, {METRICS_BASE_PORT} plus the env id by default")
    args = parser.parse_args()
    actor_id = str(args.env_id)

    configure_logging(service="actor", actor_id=actor_id)
    metrics = ActorMetrics(actor_id)
    metrics.serve(args.metrics_port if args.metrics_port is not None else METRICS_BASE_PORT + args.env_id)
    init_db()
    config = load_broker_config()
    ensure_topics(config)
    publisher = RolloutPublisher(config, args.env_id)
    policy_source = PolicySubscription(config)
    registry = GameRegistry()
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
                run_environment(connection, publisher, policy_source, metrics)
        finally:
            connection.close()
            registry.release(game_id, actor_id)
    finally:
        policy_source.close()
        publisher.close()


if __name__ == "__main__":
    main()
