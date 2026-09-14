"""An actor: plays one game instance on the learner's newest published policy, and publishes each completed rollout to the broker."""

import argparse
import logging

from broker.config import load_broker_config
from broker.policy import PolicySubscription
from broker.rollouts import RolloutPublisher
from broker.topics import ensure_topics
from db.session import init_db
from environment.run_env import run_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", type=int, required=True, help="identifies this actor in logs and on the rollout topic")
    parser.add_argument("--port", type=int, required=True, help="port of the middleman for this actor's game instance")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    init_db()
    config = load_broker_config()
    ensure_topics(config)
    publisher = RolloutPublisher(config, args.env_id)
    policy_source = PolicySubscription(config)
    try:
        run_environment(args.env_id, args.port, publisher, policy_source)
    finally:
        policy_source.close()
        publisher.close()


if __name__ == "__main__":
    main()
