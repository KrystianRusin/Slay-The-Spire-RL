"""Publishing encoded rollouts from actors, and consuming them in the learner."""

from confluent_kafka import Consumer, KafkaException, Producer

from broker.config import MAX_MESSAGE_BYTES

POLL_SECONDS = 1.0
FLUSH_SECONDS = 60.0


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


def consume_rollouts(config):
    """Yield encoded rollouts from the rollout topic as a member of the learner's consumer group.

    Runs until closed; close it to leave the group.
    """
    consumer = Consumer({
        "bootstrap.servers": config.bootstrap_servers,
        "group.id": config.learner_group,
        "auto.offset.reset": "earliest",
        "max.partition.fetch.bytes": MAX_MESSAGE_BYTES,
    })
    consumer.subscribe([config.rollout_topic])
    try:
        while True:
            message = consumer.poll(POLL_SECONDS)
            if message is None:
                continue
            error = message.error()
            if error is None:
                yield message.value()
            elif error.fatal():
                raise KafkaException(error)
            else:
                print(f"Rollout consumer: {error}")
    finally:
        consumer.close()
