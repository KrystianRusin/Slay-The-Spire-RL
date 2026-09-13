"""The learner's training loop, fed rollouts as they arrive."""

import struct

import pytest
import torch as th
from sb3_contrib.ppo_mask import MaskablePPO

import learner
from learner import train
from model.checkpoint import load_checkpoint
from model.rollout_codec import UnsupportedSchemaVersion, encode_rollout
from slay_the_spire_env import SlayTheSpireEnv

from tests.rollouts import fill, make_buffer

ROLLOUT_STEPS = 8


class Killed(BaseException):
    """Stands in for the learner process being killed at a chosen point."""


class FakeDelivery:
    """A rollout as the consumer hands it over, recording whether its offset was committed."""

    def __init__(self, value, on_commit=None):
        self.value = value
        self.committed = False
        self._on_commit = on_commit

    def commit(self):
        if self._on_commit:
            self._on_commit(self)
        self.committed = True


@pytest.fixture(autouse=True)
def in_scratch_directory(tmp_path, monkeypatch):
    """update_model appends to a log file in the working directory."""
    monkeypatch.chdir(tmp_path)


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
