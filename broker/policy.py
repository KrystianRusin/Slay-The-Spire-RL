"""Publishing policy weights from the learner, and reading the newest of them in actors.

Why the policy topic is laid out this way is recorded in docs/adr/0004.
"""

import logging
import time

from confluent_kafka import Consumer, KafkaException, TopicPartition

from broker.config import MAX_MESSAGE_BYTES
from broker.publisher import Publisher

POLICY_KEY = "policy"
POLICY_PARTITION = 0
TIMEOUT_SECONDS = 30.0

logger = logging.getLogger(__name__)


class PolicyPublisher(Publisher):
    """Publishes the learner's encoded policy weights, all under one key so compaction keeps only the newest."""

    def __init__(self, config):
        super().__init__(config, config.policy_topic, key=POLICY_KEY, client_id="learner")


class PolicySubscription:
    """Reads the newest encoded policy weights from the policy topic.

    It reads without joining a consumer group, so any number of actors can
    subscribe, and fetches nothing between reads, so versions superseded
    before the next read are never downloaded.
    """

    def __init__(self, config):
        self._topic = config.policy_topic
        self._next_offset = 0
        self._consumer = Consumer({
            "bootstrap.servers": config.bootstrap_servers,
            # Required by the client, but unused: partitions are assigned directly and no offset is ever committed.
            "group.id": "policy-subscription",
            "enable.auto.commit": False,
            "auto.offset.reset": "earliest",
            "max.partition.fetch.bytes": MAX_MESSAGE_BYTES,
        })

    def newest(self, timeout=0.0):
        """The newest weights published since the last call, or None if there are none.

        When nothing newer is on the topic yet, waits up to timeout for it.
        Broker errors are logged and read as nothing new.
        """
        try:
            return self._read_newest(timeout)
        except KafkaException as error:
            logger.warning("Could not read the policy topic: %s", error)
            return None

    def close(self):
        self._consumer.close()

    def _read_newest(self, timeout):
        _, high = self._consumer.get_watermark_offsets(
            TopicPartition(self._topic, POLICY_PARTITION), timeout=TIMEOUT_SECONDS, cached=False,
        )
        target = max(high - 1, self._next_offset)
        wait = TIMEOUT_SECONDS if high > self._next_offset else timeout
        self._consumer.assign([TopicPartition(self._topic, POLICY_PARTITION, target)])
        try:
            newest = self._poll_from(target, wait)
        finally:
            self._consumer.unassign()

        if newest is None and high > self._next_offset:
            logger.warning("Policy weights at offset %d were on the topic but not received within %ss", target, TIMEOUT_SECONDS)
        return newest

    def _poll_from(self, target, wait):
        newest = None
        deadline = time.monotonic() + wait
        while True:
            remaining = 0 if newest is not None else max(0.0, deadline - time.monotonic())
            message = self._consumer.poll(remaining)
            if message is None:
                if newest is not None or remaining == 0:
                    return newest
                continue
            error = message.error()
            if error is not None:
                if error.fatal():
                    raise KafkaException(error)
                logger.warning("Policy subscription: %s", error)
            elif message.offset() >= target:
                newest = message.value()
                self._next_offset = message.offset() + 1
