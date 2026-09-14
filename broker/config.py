"""Kafka connection details and client settings, from the environment."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Matches the rollout topic's max.message.bytes; see docs/adr/0001.
MAX_MESSAGE_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class BrokerConfig:
    bootstrap_servers: str
    rollout_topic: str = "rollouts"
    policy_topic: str = "policy"
    learner_group: str = "learner"
    replication_factor: int = 1


def load_broker_config():
    load_dotenv()
    bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
    if not bootstrap_servers:
        raise RuntimeError(
            "KAFKA_BOOTSTRAP_SERVERS is not set. Copy .env.example to .env and fill it in."
        )
    return BrokerConfig(bootstrap_servers, replication_factor=int(os.getenv("KAFKA_REPLICATION_FACTOR", "1")))
