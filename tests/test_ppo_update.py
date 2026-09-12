"""The PPO update: its per-minibatch loss, and one full update over a rollout."""

import warnings

import pytest
import torch as th
from sb3_contrib.ppo_mask import MaskablePPO

from model.model_utils import ppo_loss, update_model
from slay_the_spire_env import SlayTheSpireEnv

from tests.rollouts import fill, make_buffer


def make_model():
    return MaskablePPO("MultiInputPolicy", SlayTheSpireEnv({}), device="cpu", seed=0)


@pytest.fixture(scope="module")
def model():
    return make_model()


@pytest.fixture
def batch(model):
    return next(fill(make_buffer(model.observation_space, size=16)).get(16))


def batch_loss(model, batch, **changes):
    return ppo_loss(model.policy, dict(batch, **changes), clip_range=0.2, ent_coef=0.01, vf_coef=0.5)


def test_policy_loss_ignores_the_scale_and_offset_of_advantages(model, batch):
    advantages = batch["advantages"]

    raw = batch_loss(model, batch)
    rescaled = batch_loss(model, batch, advantages=advantages * 25.0 + 3.0)

    assert th.allclose(raw.policy, rescaled.policy, atol=1e-5)


def test_value_loss_is_the_mean_squared_error_per_step(model, batch):
    with th.no_grad():
        predicted = model.policy.predict_values(batch["observations"]).flatten()
    expected = th.mean((batch["returns"] - predicted) ** 2)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        value_loss = batch_loss(model, batch).value

    assert th.allclose(value_loss, expected, atol=1e-5)


def test_an_update_changes_the_policy_weights(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model = make_model()
    before = [parameter.detach().clone() for parameter in model.policy.parameters()]

    update_model(model, fill(make_buffer(model.observation_space, size=64)), current_step=0, total_steps=100)

    after = list(model.policy.parameters())
    assert any(not th.equal(old, new) for old, new in zip(before, after))
    assert "Error" not in (tmp_path / "model_update_log.txt").read_text()
