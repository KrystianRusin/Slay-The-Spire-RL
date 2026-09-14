"""The Kafka topics this system uses, declared here and applied by ensure_topics.

Why the rollout topic is laid out this way is recorded in docs/adr/0002, and
the policy topic in docs/adr/0004.
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
POLICY_PARTITIONS = 1
POLICY_TOPIC_CONFIG = {
    "cleanup.policy": "compact",
    "max.message.bytes": str(MAX_MESSAGE_BYTES),
    # Compaction never touches the segment being written, so this bounds how long superseded versions stay on disk.
    "segment.ms": str(60 * 60 * 1000),
}
TIMEOUT_SECONDS = 30


def ensure_topics(config):
    """Create the rollout and policy topics if they are missing, and set their configuration to match this module.

    Raises RuntimeError if either exists with a different partition count.
    """
    admin = AdminClient({"bootstrap.servers": config.bootstrap_servers})
    _ensure_topic(admin, config, config.rollout_topic, ROLLOUT_PARTITIONS, ROLLOUT_TOPIC_CONFIG)
    _ensure_topic(admin, config, config.policy_topic, POLICY_PARTITIONS, POLICY_TOPIC_CONFIG)


def _ensure_topic(admin, config, topic, declared_partitions, settings):
    new_topic = NewTopic(
        topic,
        num_partitions=declared_partitions,
        replication_factor=config.replication_factor,
        config=settings,
    )
    try:
        admin.create_topics([new_topic])[topic].result(timeout=TIMEOUT_SECONDS)
        return
    except KafkaException as error:
        if error.args[0].code() != KafkaError.TOPIC_ALREADY_EXISTS:
            raise

    partitions = _partition_count(admin, topic)
    if partitions != declared_partitions:
        raise RuntimeError(
            f"Topic {topic} has {partitions} partitions but {declared_partitions} are declared; "
            "repartition it deliberately or update its declaration in broker/topics.py"
        )
    entries = [
        ConfigEntry(name, value, incremental_operation=AlterConfigOpType.SET)
        for name, value in settings.items()
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

