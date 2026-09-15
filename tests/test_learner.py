"""The learner's training loop, fed rollouts as they arrive."""

import struct

import pytest
import torch as th
from confluent_kafka import KafkaException
from sb3_contrib.ppo_mask import MaskablePPO

import learner
from learner import check_lag, train
from model.checkpoint import load_checkpoint
from model.policy_codec import decode_policy
from model.rollout_codec import UnsupportedSchemaVersion, encode_rollout
from observability.metrics import LearnerMetrics
from slay_the_spire_env import SlayTheSpireEnv

from tests.rollouts import fill, make_buffer

ROLLOUT_STEPS = 8


class Killed(BaseException):
    """Stands in for the learner process being killed at a chosen point."""


class FakeDelivery:
    """A rollout as the consumer hands it over, recording whether its offset was committed."""

    def __init__(self, value, on_commit=None, published_at=None):
        self.value = value
        self.published_at = published_at
        self.committed = False
        self._on_commit = on_commit

    def commit(self):
        if self._on_commit:
            self._on_commit(self)
        self.committed = True


class FakePublisher:
    """The policy topic as the learner writes to it."""

    def __init__(self):
        self.published = []

    def publish(self, value):
        self.published.append(decode_policy(value))

    def versions(self):
        return [policy.version for policy in self.published]


def make_model():
    return MaskablePPO("MultiInputPolicy", SlayTheSpireEnv({}), device="cpu", seed=0)


def encoded_rollout(model, rollout_id=None):
    return encode_rollout(fill(make_buffer(model.observation_space, size=ROLLOUT_STEPS)), rollout_id)


def weights(model):
    return [parameter.detach().clone() for parameter in model.policy.parameters()]


def restart(save_path):
    return load_checkpoint(save_path, SlayTheSpireEnv({}), device="cpu")


def test_each_rollout_is_applied_before_the_next_is_awaited(tmp_path):
    model = make_model()
    seen_before_each_read = []

    def arriving():
        for _ in range(2):
            seen_before_each_read.append(weights(model))
            yield FakeDelivery(encoded_rollout(model))

    train(model, arriving(), total_steps=2 * ROLLOUT_STEPS, save_path=tmp_path / "policy")

    first, second = seen_before_each_read
    assert any(not th.equal(old, new) for old, new in zip(first, second))


def test_training_stops_once_total_steps_are_consumed(tmp_path):
    model = make_model()
    delivered = []

    def arriving():
        while True:
            delivered.append(FakeDelivery(encoded_rollout(model)))
            yield delivered[-1]

    steps = train(model, arriving(), total_steps=2 * ROLLOUT_STEPS, save_path=tmp_path / "policy")

    assert steps == 2 * ROLLOUT_STEPS
    assert len(delivered) == 2


def test_a_rollout_is_committed_only_once_its_update_is_saved(tmp_path):
    model = make_model()
    save_path = tmp_path / "policy"
    saved_at_commit = []

    def check_saved(_delivery):
        _, progress = restart(save_path)
        saved_at_commit.append(progress.has_applied("rollout-a"))

    delivery = FakeDelivery(encoded_rollout(model, "rollout-a"), on_commit=check_saved)
    train(model, [delivery], total_steps=ROLLOUT_STEPS, save_path=save_path)

    assert delivery.committed
    assert saved_at_commit == [True]


def test_a_learner_killed_mid_update_reapplies_the_uncommitted_rollout_on_restart(tmp_path, monkeypatch):
    model = make_model()
    save_path = tmp_path / "policy"
    first = FakeDelivery(encoded_rollout(model, "rollout-a"))
    second = FakeDelivery(encoded_rollout(model, "rollout-b"))
    real_update = learner.update_model

    def killed_during_second_update(model, rollout, *args):
        if rollout.rollout_id == "rollout-b":
            raise Killed
        real_update(model, rollout, *args)

    monkeypatch.setattr(learner, "update_model", killed_during_second_update)
    with pytest.raises(Killed):
        train(model, [first, second], total_steps=10 * ROLLOUT_STEPS, save_path=save_path)
    monkeypatch.setattr(learner, "update_model", real_update)
    assert first.committed and not second.committed

    model, progress = restart(save_path)
    redelivered = FakeDelivery(second.value)
    steps = train(model, [redelivered], total_steps=10 * ROLLOUT_STEPS, progress=progress, save_path=save_path)

    assert redelivered.committed
    assert steps == 2 * ROLLOUT_STEPS
    assert restart(save_path)[1].has_applied("rollout-b")


def test_a_rollout_redelivered_after_its_update_was_saved_is_not_applied_again(tmp_path):
    model = make_model()
    save_path = tmp_path / "policy"

    def killed_before_commit(_delivery):
        raise Killed

    applied = FakeDelivery(encoded_rollout(model, "rollout-a"), on_commit=killed_before_commit)
    with pytest.raises(Killed):
        train(model, [applied], total_steps=10 * ROLLOUT_STEPS, save_path=save_path)

    model, progress = restart(save_path)
    after_first_update = weights(model)
    redelivered = FakeDelivery(applied.value)
    steps = train(model, [redelivered], total_steps=10 * ROLLOUT_STEPS, progress=progress, save_path=save_path)

    assert redelivered.committed
    assert steps == ROLLOUT_STEPS
    assert all(th.equal(old, new) for old, new in zip(after_first_update, model.policy.parameters()))


def test_a_rollout_from_a_newer_schema_stops_the_learner_without_committing_it(tmp_path):
    model = make_model()
    newer = bytearray(encoded_rollout(model))
    struct.pack_into("<H", newer, 4, 99)
    delivery = FakeDelivery(bytes(newer))

    with pytest.raises(UnsupportedSchemaVersion, match="99"):
        train(model, [delivery], total_steps=ROLLOUT_STEPS, save_path=tmp_path / "policy")

    assert not delivery.committed


def test_an_undecodable_rollout_is_committed_and_training_continues(tmp_path):
    model = make_model()
    poison = FakeDelivery(b"not a rollout")
    good = FakeDelivery(encoded_rollout(model))

    steps = train(model, [poison, good], total_steps=ROLLOUT_STEPS, save_path=tmp_path / "policy")

    assert poison.committed and good.committed
    assert steps == ROLLOUT_STEPS


def test_each_update_is_published_as_the_next_version_before_its_rollout_is_committed(tmp_path):
    model = make_model()
    publisher = FakePublisher()
    published_at_commit = []

    def check_published(_delivery):
        published_at_commit.append(publisher.versions()[-1])

    deliveries = [FakeDelivery(encoded_rollout(model), on_commit=check_published) for _ in range(2)]
    train(model, deliveries, total_steps=2 * ROLLOUT_STEPS, save_path=tmp_path / "policy", publisher=publisher)

    assert publisher.versions() == [0, 1, 2]
    assert published_at_commit == [1, 2]
    newest = make_model()
    publisher.published[-1].apply_to(newest.policy)
    assert all(th.equal(old, new) for old, new in zip(weights(model), newest.policy.parameters()))


def test_rollouts_that_are_skipped_publish_no_new_version(tmp_path):
    model = make_model()
    save_path = tmp_path / "policy"
    applied = encoded_rollout(model, "rollout-a")
    train(model, [FakeDelivery(applied)], total_steps=10 * ROLLOUT_STEPS, save_path=save_path)

    model, progress = restart(save_path)
    publisher = FakePublisher()
    train(model, [FakeDelivery(applied), FakeDelivery(b"not a rollout")], total_steps=10 * ROLLOUT_STEPS,
          progress=progress, save_path=save_path, publisher=publisher)

    assert publisher.versions() == [1]


def test_each_update_records_the_age_of_its_rollout_and_the_new_policy_version(tmp_path):
    model = make_model()
    metrics = LearnerMetrics(clock=lambda: 1000.0)
    deliveries = [FakeDelivery(encoded_rollout(model), published_at=published_at) for published_at in (940.0, 990.0)]

    train(model, deliveries, total_steps=2 * ROLLOUT_STEPS, save_path=tmp_path / "policy", metrics=metrics)

    assert metrics.registry.get_sample_value("sts_learner_rollout_age_seconds_count") == 2
    assert metrics.registry.get_sample_value("sts_learner_rollout_age_seconds_sum") == 70.0
    assert metrics.registry.get_sample_value("sts_learner_last_rollout_age_seconds") == 10.0
    assert metrics.registry.get_sample_value("sts_learner_updates_total") == 2
    assert metrics.registry.get_sample_value("sts_learner_policy_version") == 2
    assert metrics.registry.get_sample_value("sts_learner_steps") == 2 * ROLLOUT_STEPS


def test_skipped_rollouts_are_counted_by_why_they_were_skipped(tmp_path):
    model = make_model()
    save_path = tmp_path / "policy"
    applied = encoded_rollout(model, "rollout-a")
    train(model, [FakeDelivery(applied)], total_steps=10 * ROLLOUT_STEPS, save_path=save_path)

    model, progress = restart(save_path)
    metrics = LearnerMetrics()
    train(model, [FakeDelivery(applied), FakeDelivery(b"not a rollout")], total_steps=10 * ROLLOUT_STEPS,
          progress=progress, save_path=save_path, metrics=metrics)

    sample = metrics.registry.get_sample_value
    assert sample("sts_learner_rollouts_skipped_total", {"reason": "already_applied"}) == 1
    assert sample("sts_learner_rollouts_skipped_total", {"reason": "undecodable"}) == 1
    assert sample("sts_learner_updates_total") == 0
    assert sample("sts_learner_policy_version") == 1


class FakeLagProbe:
    """Returns each measurement in turn, raising those that are exceptions."""

    def __init__(self, *measurements):
        self._measurements = iter(measurements)

    def measure(self):
        measurement = next(self._measurements)
        if isinstance(measurement, Exception):
            raise measurement
        return measurement


def test_consumer_lag_is_exported_by_partition_warned_about_once_rollouts_pile_up_and_dropped_when_unmeasurable(caplog):
    metrics = LearnerMetrics()
    probe = FakeLagProbe({0: 1, 1: 2}, {0: 3, 1: 4}, KafkaException("broker unreachable"))
    sample = metrics.registry.get_sample_value

    check_lag(probe, metrics)
    assert sample("sts_learner_consumer_lag_rollouts", {"partition": "1"}) == 2
    assert not [record for record in caplog.records if record.levelname == "WARNING"]

    check_lag(probe, metrics)
    assert sample("sts_learner_consumer_lag_rollouts", {"partition": "0"}) == 3
    assert "7 rollouts behind" in caplog.text

    check_lag(probe, metrics)
    assert "Could not measure consumer lag" in caplog.text
    assert sample("sts_learner_consumer_lag_rollouts", {"partition": "1"}) is None
