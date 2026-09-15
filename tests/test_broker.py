"""Rollouts and policy weights over a real Kafka broker: topic setup, actors publishing to the learner, and the learner publishing to actors."""

import multiprocessing
import os
import time
import uuid
from contextlib import closing

import numpy as np
import pytest
import torch as th
from confluent_kafka import KafkaException
from confluent_kafka.admin import AdminClient, ConfigResource, NewTopic, ResourceType

from broker.config import BrokerConfig
from broker.policy import PolicyPublisher, PolicySubscription
from broker.rollouts import ConsumerLagProbe, RolloutConsumer, RolloutPublisher
from broker.topics import POLICY_PARTITIONS, POLICY_TOPIC_CONFIG, ROLLOUT_PARTITIONS, ROLLOUT_TOPIC_CONFIG, ensure_topics
from learner import train
from model.policy_codec import encode_policy
from model.policy_follower import PolicyFollower
from model.rollout_codec import decode_rollout, encode_rollout

from tests.rollouts import ACTION_SPACE, fill, make_buffer
from tests.test_learner import make_model, weights
from tests.test_policy_follower import learner_at, runs

ROLLOUT_STEPS = 8
ROLLOUTS_PER_ACTOR = 2


@pytest.fixture
def broker():
    bootstrap_servers = os.environ.get("TEST_KAFKA_BOOTSTRAP_SERVERS")
    if not bootstrap_servers:
        pytest.skip("TEST_KAFKA_BOOTSTRAP_SERVERS is not set")
    suffix = uuid.uuid4().hex[:8]
    config = BrokerConfig(
        bootstrap_servers,
        rollout_topic=f"rollouts-test-{suffix}",
        policy_topic=f"policy-test-{suffix}",
        learner_group=f"learner-test-{suffix}",
    )
    yield config
    admin = AdminClient({"bootstrap.servers": bootstrap_servers})
    topics = admin.list_topics(timeout=30).topics
    created = [topic for topic in (config.rollout_topic, config.policy_topic) if topic in topics]
    if created:
        for future in admin.delete_topics(created).values():
            future.result(timeout=30)


def settled_topic_state(config, topic, expected_configs):
    """Partition count and configs, once the broker reports expected_configs or 30s pass.

    The broker applies topic changes shortly after the admin call that made them returns.
    """
    admin = AdminClient({"bootstrap.servers": config.bootstrap_servers})
    resource = ConfigResource(ResourceType.TOPIC, topic)
    deadline = time.monotonic() + 30
    while True:
        try:
            partitions = admin.list_topics(topic, timeout=30).topics[topic].partitions
            entries = admin.describe_configs([resource])[resource].result(timeout=30)
            configs = {name: entry.value for name, entry in entries.items()}
            if expected_configs.items() <= configs.items():
                return len(partitions), configs
        except KafkaException:
            if time.monotonic() > deadline:
                raise
        if time.monotonic() > deadline:
            return len(partitions), configs
        time.sleep(0.2)


def actor_rollout(observation_space, actor_id):
    buffer = fill(make_buffer(observation_space, size=ROLLOUT_STEPS))
    buffer.rewards[:] = actor_id
    return encode_rollout(buffer)


def publish_as_actor(config, actor_id, observation_space):
    publisher = RolloutPublisher(config, actor_id)
    try:
        for _ in range(ROLLOUTS_PER_ACTOR):
            publisher.publish(actor_rollout(observation_space, actor_id))
    finally:
        publisher.close()


def publish_once_then_keep_playing(config, actor_id, observation_space, published):
    publisher = RolloutPublisher(config, actor_id)
    publisher.publish(actor_rollout(observation_space, actor_id))
    published.set()
    time.sleep(600)


def publish(config, *encoded):
    publisher = RolloutPublisher(config, actor_id=0)
    for value in encoded:
        publisher.publish(value)
    publisher.close()


def start_actors(config, count, observation_space, first_id=0):
    context = multiprocessing.get_context("spawn")
    actors = [context.Process(target=publish_as_actor, args=(config, actor_id, observation_space))
              for actor_id in range(first_id, first_id + count)]
    for actor in actors:
        actor.start()
    return actors


def join(actors):
    for actor in actors:
        actor.join(timeout=120)
        assert actor.exitcode == 0


def test_the_rollout_topic_is_created_as_specified(broker):
    ensure_topics(broker)

    partitions, configs = settled_topic_state(broker, broker.rollout_topic, ROLLOUT_TOPIC_CONFIG)

    assert partitions == ROLLOUT_PARTITIONS
    for name, value in ROLLOUT_TOPIC_CONFIG.items():
        assert configs[name] == value, name


def test_the_policy_topic_is_created_compacted(broker):
    ensure_topics(broker)

    partitions, configs = settled_topic_state(broker, broker.policy_topic, POLICY_TOPIC_CONFIG)

    assert partitions == POLICY_PARTITIONS
    for name, value in POLICY_TOPIC_CONFIG.items():
        assert configs[name] == value, name


def test_topic_settings_changed_by_hand_are_restored(broker):
    admin = AdminClient({"bootstrap.servers": broker.bootstrap_servers})
    drifted = NewTopic(broker.rollout_topic, num_partitions=ROLLOUT_PARTITIONS, replication_factor=1, config={"retention.ms": "1000"})
    admin.create_topics([drifted])[broker.rollout_topic].result(timeout=30)

    ensure_topics(broker)

    assert settled_topic_state(broker, broker.rollout_topic, ROLLOUT_TOPIC_CONFIG)[1]["retention.ms"] == ROLLOUT_TOPIC_CONFIG["retention.ms"]


def test_a_topic_with_the_wrong_partition_count_is_refused(broker):
    admin = AdminClient({"bootstrap.servers": broker.bootstrap_servers})
    admin.create_topics([NewTopic(broker.rollout_topic, num_partitions=1, replication_factor=1)])[broker.rollout_topic].result(timeout=30)

    with pytest.raises(RuntimeError, match="partitions"):
        ensure_topics(broker)


def test_rollouts_from_separate_actor_processes_all_reach_the_learner(broker, observation_space):
    ensure_topics(broker)
    actors = start_actors(broker, 3, observation_space)

    received = []
    with closing(RolloutConsumer(broker)) as deliveries:
        for delivery in deliveries:
            received.append(decode_rollout(delivery.value, observation_space, ACTION_SPACE))
            if len(received) == 3 * ROLLOUTS_PER_ACTOR:
                break
    join(actors)

    senders = sorted(int(rollout.rewards[0]) for rollout in received)
    assert senders == sorted(list(range(3)) * ROLLOUTS_PER_ACTOR)


def test_an_incompressible_full_size_rollout_fits_on_the_topic(broker, observation_space):
    ensure_topics(broker)
    buffer = make_buffer(observation_space, size=2048)
    generator = np.random.default_rng(0)
    for observations in buffer.observations.values():
        observations[:] = generator.random(observations.shape, dtype=np.float32)
    buffer.pos, buffer.full = 2048, True
    encoded = encode_rollout(buffer)
    assert len(encoded) > 10 * 1024 * 1024

    publish(broker, encoded)

    with closing(RolloutConsumer(broker)) as deliveries:
        assert next(iter(deliveries)).value == encoded


def test_the_learner_trains_on_however_many_actors_are_publishing(broker, observation_space, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ensure_topics(broker)
    model = make_model()
    before = weights(model)
    actors = start_actors(broker, 3, observation_space)

    with closing(RolloutConsumer(broker)) as deliveries:
        steps = train(model, deliveries, total_steps=3 * ROLLOUTS_PER_ACTOR * ROLLOUT_STEPS)
    join(actors)

    assert steps == 3 * ROLLOUTS_PER_ACTOR * ROLLOUT_STEPS
    assert any(not th.equal(old, new) for old, new in zip(before, model.policy.parameters()))


def test_a_rollout_left_uncommitted_is_delivered_to_the_next_learner_in_the_group(broker):
    ensure_topics(broker)
    publish(broker, b"first", b"second")

    with closing(RolloutConsumer(broker)) as deliveries:
        received = iter(deliveries)
        next(received).commit()
        assert next(received).value == b"second"

    with closing(RolloutConsumer(broker)) as deliveries:
        assert next(iter(deliveries)).value == b"second"


def test_lag_counts_the_rollouts_the_learner_group_has_not_committed(broker):
    ensure_topics(broker)

    with closing(ConsumerLagProbe(broker)) as probe:
        assert sorted(probe.measure()) == list(range(ROLLOUT_PARTITIONS))
        assert sum(probe.measure().values()) == 0

        publish(broker, b"first", b"second", b"third")
        assert sum(probe.measure().values()) == 3

        with closing(RolloutConsumer(broker)) as deliveries:
            next(iter(deliveries)).commit()
            assert sum(probe.measure().values()) == 2


def test_a_delivery_carries_when_its_rollout_was_published(broker):
    ensure_topics(broker)
    before = time.time()
    publish(broker, b"rollout")
    after = time.time()

    with closing(RolloutConsumer(broker)) as deliveries:
        published_at = next(iter(deliveries)).published_at

    assert before - 1 <= published_at <= after + 1


def test_the_learner_keeps_training_after_an_actor_is_killed(broker, observation_space, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ensure_topics(broker)
    context = multiprocessing.get_context("spawn")
    published = context.Event()
    doomed = context.Process(target=publish_once_then_keep_playing, args=(broker, 0, observation_space, published))
    doomed.start()
    assert published.wait(timeout=120)
    doomed.kill()
    doomed.join()
    survivors = start_actors(broker, 2, observation_space, first_id=1)

    with closing(RolloutConsumer(broker)) as deliveries:
        steps = train(make_model(), deliveries, total_steps=(1 + 2 * ROLLOUTS_PER_ACTOR) * ROLLOUT_STEPS)
    join(survivors)

    assert steps == (1 + 2 * ROLLOUTS_PER_ACTOR) * ROLLOUT_STEPS


def publish_versions(config, versions):
    with closing(PolicyPublisher(config)) as publisher:
        for version in versions:
            newest = learner_at(version)
            publisher.publish(encode_policy(newest.policy, version=version))
    return newest


def test_an_actor_started_after_several_versions_runs_the_newest(broker):
    ensure_topics(broker)
    newest = publish_versions(broker, [1, 2, 3])

    with closing(PolicySubscription(broker)) as subscription:
        follower = PolicyFollower(make_model().policy, subscription)
        follower.wait_for_first()

    assert follower.version == 3
    assert runs(follower, newest)


def test_an_actor_that_waits_for_the_first_version_gets_it_once_published(broker):
    ensure_topics(broker)
    context = multiprocessing.get_context("spawn")
    learner = context.Process(target=publish_versions, args=(broker, [1]))

    with closing(PolicySubscription(broker)) as subscription:
        assert subscription.newest(timeout=1.0) is None
        learner.start()
        follower = PolicyFollower(make_model().policy, subscription)
        follower.wait_for_first()
    learner.join(timeout=120)

    assert follower.version == 1


def test_an_update_skips_versions_superseded_since_the_last(broker):
    ensure_topics(broker)
    publish_versions(broker, [1])

    with closing(PolicySubscription(broker)) as subscription:
        follower = PolicyFollower(make_model().policy, subscription)
        follower.wait_for_first()
        newest = publish_versions(broker, [2, 3, 4, 5])

        assert follower.update()
        assert not follower.update()

    assert follower.version == 5
    assert runs(follower, newest)


def test_the_learner_publishes_each_update_for_actors(broker, observation_space, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ensure_topics(broker)
    model = make_model()
    publish(broker, actor_rollout(observation_space, 0), actor_rollout(observation_space, 1))

    with closing(PolicyPublisher(broker)) as publisher, closing(RolloutConsumer(broker)) as deliveries:
        train(model, deliveries, total_steps=2 * ROLLOUT_STEPS, publisher=publisher)

    with closing(PolicySubscription(broker)) as subscription:
        follower = PolicyFollower(make_model().policy, subscription)
        follower.wait_for_first()

    assert follower.version == 2
    assert all(th.equal(old, new) for old, new in zip(weights(model), follower.policy.parameters()))
