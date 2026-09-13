"""The learner's training loop, fed rollouts as they arrive."""

import torch as th
from sb3_contrib.ppo_mask import MaskablePPO

from learner import train
from model.rollout_codec import encode_rollout
from slay_the_spire_env import SlayTheSpireEnv

from tests.rollouts import fill, make_buffer

ROLLOUT_STEPS = 8


def make_model():
    return MaskablePPO("MultiInputPolicy", SlayTheSpireEnv({}), device="cpu", seed=0)


def encoded_rollout(model):
    return encode_rollout(fill(make_buffer(model.observation_space, size=ROLLOUT_STEPS)))


def weights(model):
    return [parameter.detach().clone() for parameter in model.policy.parameters()]


def test_each_rollout_is_applied_before_the_next_is_awaited(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model = make_model()
    seen_before_each_read = []

    def arriving():
        for _ in range(2):
            seen_before_each_read.append(weights(model))
            yield encoded_rollout(model)

    train(model, arriving(), total_steps=2 * ROLLOUT_STEPS)

    first, second = seen_before_each_read
    assert any(not th.equal(old, new) for old, new in zip(first, second))


def test_training_stops_once_total_steps_are_consumed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model = make_model()
    delivered = []

    def arriving():
        while True:
            delivered.append(encoded_rollout(model))
            yield delivered[-1]

    steps = train(model, arriving(), total_steps=2 * ROLLOUT_STEPS)

    assert steps == 2 * ROLLOUT_STEPS
    assert len(delivered) == 2


def test_the_model_is_saved_after_each_update(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model = make_model()

    train(model, [encoded_rollout(model)], total_steps=ROLLOUT_STEPS, save_path=tmp_path / "policy")

    assert (tmp_path / "policy.zip").exists()
