"""An actor: plays one game instance and publishes each completed rollout to the broker."""

import argparse

from broker.config import load_broker_config
from broker.rollouts import RolloutPublisher
from broker.topics import ensure_topics
from db.session import init_db
from environment.run_env import run_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", type=int, required=True, help="identifies this actor in logs and on the rollout topic")
    parser.add_argument("--port", type=int, required=True, help="port of the middleman for this actor's game instance")
    args = parser.parse_args()

    init_db()
    config = load_broker_config()
    ensure_topics(config)
    publisher = RolloutPublisher(config, args.env_id)
    try:
        run_environment(args.env_id, args.port, publisher)
    finally:
        publisher.close()


if __name__ == "__main__":
    main()
