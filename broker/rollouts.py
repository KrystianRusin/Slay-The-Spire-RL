"""Publishing encoded rollouts from actors, and consuming them in the learner."""

import logging
from functools import partial

from confluent_kafka import Consumer, KafkaException, Producer

from broker.config import MAX_MESSAGE_BYTES

POLL_SECONDS = 1.0
FLUSH_SECONDS = 60.0
TIMEOUT_SECONDS = 30.0
# Each rollout waiting is one more update the policy moves on before that rollout is used; see docs/adr/0003.
LAG_WARNING_ROLLOUTS = 5

logger = logging.getLogger(__name__)


class RolloutPublisher:
    """Publishes one actor's rollouts to the rollout topic, keyed by actor id."""

    def __init__(self, config, actor_id):
        self.topic = config.rollout_topic
        self.key = str(actor_id)
        self._producer = Producer({
            "bootstrap.servers": config.bootstrap_servers,
            "client.id": f"actor-{actor_id}",
            "message.max.bytes": MAX_MESSAGE_BYTES,
            # Rollouts are already zlib-compressed by the codec.
            "compression.type": "none",
            # The Java client's partitioner, so producers in any language agree on each key's partition.
            "partitioner": "murmur2_random",
        })

    def publish(self, encoded):
        """Block until the broker has the rollout, raising KafkaException if delivery fails."""
        failures = []

        def on_delivery(error, _message):
            if error is not None:
                failures.append(error)

        self._producer.produce(self.topic, value=encoded, key=self.key, on_delivery=on_delivery)
        remaining = self._producer.flush(FLUSH_SECONDS)
        if failures:
            raise KafkaException(failures[0])
        if remaining:
            raise KafkaException(f"Rollout was not delivered within {FLUSH_SECONDS}s")

    def close(self):
        self._producer.flush(FLUSH_SECONDS)


class Delivery:
    """One rollout read from the topic. Its offset is committed only when commit is called."""

    def __init__(self, value, commit):
        self.value = value
        self.commit = commit


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
                yield Delivery(message.value(), partial(self._commit, message))
            elif error.fatal():
                raise KafkaException(error)
            else:
                logger.warning("Rollout consumer: %s", error)

    def lag(self):
        """Rollouts on this member's partitions that are on the topic but not yet committed."""
        assignment = self._consumer.assignment()
        if not assignment:
            return 0
        total = 0
        for partition in self._consumer.committed(assignment, timeout=TIMEOUT_SECONDS):
            low, high = self._consumer.get_watermark_offsets(partition, timeout=TIMEOUT_SECONDS, cached=False)
            total += high - (partition.offset if partition.offset >= 0 else low)
        return total

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
            return
        self._report_lag()

    def _report_lag(self):
        try:
            lag = self.lag()
        except KafkaException as error:
            logger.warning("Could not measure consumer lag: %s", error)
            return
        if lag >= LAG_WARNING_ROLLOUTS:
            logger.warning("Learner is %d rollouts behind the actors; the rollouts it trains on are getting stale", lag)
        else:
            logger.info("Consumer lag: %d rollouts", lag)


def _partition_numbers(partitions):
    return sorted(partition.partition for partition in partitions)
