"""Publishing messages to a topic, each confirmed by the broker before publish returns."""

from confluent_kafka import KafkaException, Producer

from broker.config import MAX_MESSAGE_BYTES

FLUSH_SECONDS = 60.0


class Publisher:
    """Publishes messages to one topic under one key."""

    def __init__(self, config, topic, key, client_id):
        self.topic = topic
        self.key = key
        self._producer = Producer({
            "bootstrap.servers": config.bootstrap_servers,
            "client.id": client_id,
            "message.max.bytes": MAX_MESSAGE_BYTES,
            # Payloads are already compressed or incompressible, so the size checked against limits is the size on the wire.
            "compression.type": "none",
            # The Java client's partitioner, so producers in any language agree on each key's partition.
            "partitioner": "murmur2_random",
        })

    def publish(self, value):
        """Block until the broker has the message, raising KafkaException if delivery fails."""
        failures = []

        def on_delivery(error, _message):
            if error is not None:
                failures.append(error)

        self._producer.produce(self.topic, value=value, key=self.key, on_delivery=on_delivery)
        remaining = self._producer.flush(FLUSH_SECONDS)
        if failures:
            raise KafkaException(failures[0])
        if remaining:
            raise KafkaException(f"Message was not delivered to {self.topic} within {FLUSH_SECONDS}s")

    def close(self):
        self._producer.flush(FLUSH_SECONDS)
