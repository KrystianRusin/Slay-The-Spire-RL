"""The Kafka topics this system uses, declared here and applied by ensure_topics.

Why the rollout topic is laid out this way is recorded in docs/adr/0002.
"""

import time

from confluent_kafka import KafkaError, KafkaException
from confluent_kafka.admin import AdminClient, AlterConfigOpType, ConfigEntry, ConfigResource, NewTopic, ResourceType

from broker.config import MAX_MESSAGE_BYTES

ROLLOUT_PARTITIONS = 6
ROLLOUT_TOPIC_CONFIG = {
    "max.message.bytes": str(MAX_MESSAGE_BYTES),
    "retention.ms": str(24 * 60 * 60 * 1000),
}
TIMEOUT_SECONDS = 30


def ensure_topics(config):
    """Create the rollout topic if it is missing, and set its configuration to match this module.

    Raises RuntimeError if the topic exists with a different partition count.
    """
    admin = AdminClient({"bootstrap.servers": config.bootstrap_servers})
    topic = config.rollout_topic
    new_topic = NewTopic(
        topic,
        num_partitions=ROLLOUT_PARTITIONS,
        replication_factor=config.replication_factor,
        config=ROLLOUT_TOPIC_CONFIG,
    )
    try:
        admin.create_topics([new_topic])[topic].result(timeout=TIMEOUT_SECONDS)
        return
    except KafkaException as error:
        if error.args[0].code() != KafkaError.TOPIC_ALREADY_EXISTS:
            raise

    partitions = _partition_count(admin, topic)
    if partitions != ROLLOUT_PARTITIONS:
        raise RuntimeError(
            f"Topic {topic} has {partitions} partitions but {ROLLOUT_PARTITIONS} are declared; "
            "repartition it deliberately or update ROLLOUT_PARTITIONS"
        )
    entries = [
        ConfigEntry(name, value, incremental_operation=AlterConfigOpType.SET)
        for name, value in ROLLOUT_TOPIC_CONFIG.items()
    ]
    resource = ConfigResource(ResourceType.TOPIC, topic, incremental_configs=entries)
    admin.incremental_alter_configs([resource])[resource].result(timeout=TIMEOUT_SECONDS)


def _partition_count(admin, topic):
    """Partition count from cluster metadata, which can lag briefly behind a topic just created elsewhere."""
    deadline = time.monotonic() + TIMEOUT_SECONDS
    while True:
        partitions = admin.list_topics(topic, timeout=TIMEOUT_SECONDS).topics[topic].partitions
        if partitions or time.monotonic() > deadline:
            return len(partitions)
        time.sleep(0.2)

