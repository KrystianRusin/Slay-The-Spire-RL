"""Publishing encoded rollouts from actors, and consuming them in the learner."""

import logging
from functools import partial

from confluent_kafka import TIMESTAMP_NOT_AVAILABLE, Consumer, KafkaException, TopicPartition

from broker.config import MAX_MESSAGE_BYTES
from broker.publisher import Publisher

POLL_SECONDS = 1.0
TIMEOUT_SECONDS = 30.0

logger = logging.getLogger(__name__)


class RolloutPublisher(Publisher):
    """Publishes one actor's rollouts to the rollout topic, keyed by actor id."""

    def __init__(self, config, actor_id):
        super().__init__(config, config.rollout_topic, key=str(actor_id), client_id=f"actor-{actor_id}")


class Delivery:
    """One rollout read from the topic. Its offset is committed only when commit is called.

    published_at is when the actor published it, in seconds since the epoch, or None if the broker did not record it.
    """

    def __init__(self, value, commit, published_at=None):
        self.value = value
        self.commit = commit
        self.published_at = published_at


class RolloutConsumer:
    """Reads rollouts from the rollout topic as a member of the learner's consumer group.

    Iterating yields a Delivery for each rollout, and runs until closed. A
    rollout whose delivery is not committed is delivered again after a restart
    or rebalance, so the reader must tolerate seeing one twice. Close the
    consumer to leave the group.
    """

    def __init__(self, config):
        self._consumer = Consumer({
            "bootstrap.servers": config.bootstrap_servers,
            "group.id": config.learner_group,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
            "max.partition.fetch.bytes": MAX_MESSAGE_BYTES,
        })
        self._consumer.subscribe(
            [config.rollout_topic],
            on_assign=lambda _consumer, partitions: logger.info("Assigned partitions %s", _partition_numbers(partitions)),
            on_revoke=lambda _consumer, partitions: logger.info("Revoked partitions %s", _partition_numbers(partitions)),
            on_lost=lambda _consumer, partitions: logger.warning("Lost partitions %s", _partition_numbers(partitions)),
        )

    def __iter__(self):
        while True:
            message = self._consumer.poll(POLL_SECONDS)
            if message is None:
                continue
            error = message.error()
            if error is None:
                yield Delivery(message.value(), partial(self._commit, message), _published_at(message))
            elif error.fatal():
                raise KafkaException(error)
            else:
                logger.warning("Rollout consumer: %s", error)

    def close(self):
        self._consumer.close()

    def _commit(self, message):
        try:
            self._consumer.commit(message=message, asynchronous=False)
        except KafkaException as error:
            logger.warning(
                "Could not commit offset %d on partition %d, so that rollout may be delivered again: %s",
                message.offset(), message.partition(), error,
            )


class ConsumerLagProbe:
    """Measures the learner group's lag on the rollout topic from outside the group.

    It never joins the group or commits, so it can run on another thread while
    the learner consumes.
    """

    def __init__(self, config):
        self._topic = config.rollout_topic
        self._consumer = Consumer({
            "bootstrap.servers": config.bootstrap_servers,
            "group.id": config.learner_group,
            "enable.auto.commit": False,
        })

    def measure(self):
        """Rollouts on the topic the learner group has not committed, by partition number."""
        metadata = self._consumer.list_topics(self._topic, timeout=TIMEOUT_SECONDS)
        partitions = [TopicPartition(self._topic, number) for number in metadata.topics[self._topic].partitions]
        lag = {}
        for partition in self._consumer.committed(partitions, timeout=TIMEOUT_SECONDS):
            low, high = self._consumer.get_watermark_offsets(partition, timeout=TIMEOUT_SECONDS, cached=False)
            lag[partition.partition] = partition_lag(partition.offset, low, high)
        return lag

    def close(self):
        self._consumer.close()


def partition_lag(committed, low, high):
    """Messages on a partition after the committed offset, given its low and high watermarks.

    committed is negative when nothing has been committed. Messages already
    removed by retention are not counted.
    """
    return high - max(committed, low)


def _published_at(message):
    kind, milliseconds = message.timestamp()
    return None if kind == TIMESTAMP_NOT_AVAILABLE else milliseconds / 1000


def _partition_numbers(partitions):
    return sorted(partition.partition for partition in partitions)
