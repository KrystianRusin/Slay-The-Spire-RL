"""The rollout buffer: array shapes, GAE, and minibatch sampling."""

import numpy as np
import pytest
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer

from tests.rollouts import GAE_LAMBDA, GAMMA, add_step, fill, make_buffer

SCALAR_ARRAYS = ["rewards", "dones", "values", "old_log_prob", "advantages", "returns"]
SINGLE_FEATURE = spaces.Dict({"x": spaces.Box(0, 1, (1,))})


def shapes(buffer):
    found = {name: getattr(buffer, name).shape for name in SCALAR_ARRAYS + ["actions"]}
    found.update({f"observations.{key}": obs.shape for key, obs in buffer.observations.items()})
    return found


def test_every_array_keeps_its_shape_across_fill_flush_and_refill(observation_space):
    buffer = make_buffer(observation_space)
    constructed = shapes(buffer)

    fill(buffer)
    assert shapes(buffer) == constructed
    buffer.reset()
    assert shapes(buffer) == constructed
    fill(buffer)
    assert shapes(buffer) == constructed


def test_arrays_are_sized_per_step(observation_space):
    buffer = make_buffer(observation_space, size=8)

    for name in SCALAR_ARRAYS:
        assert getattr(buffer, name).shape == (8,), name
    for key, space in observation_space.spaces.items():
        assert buffer.observations[key].shape == (8, *space.shape), key


@pytest.mark.parametrize("rollouts", [1, 2, 3])
def test_returns_are_one_dimensional_on_every_rollout(observation_space, rollouts):
    buffer = make_buffer(observation_space, size=8)
    fill(buffer)
    for _ in range(rollouts - 1):
        buffer.reset()
        fill(buffer)

    assert buffer.returns.shape == (8,)
    assert buffer.advantages.shape == (8,)


def test_adding_does_not_write_returns(observation_space):
    buffer = make_buffer(observation_space, size=4)
    add_step(buffer, value=5.0)

    assert buffer.returns[0] == 0.0


REWARDS = [1.0, 0.0, 2.0, -1.0, 0.5, 0.25]
VALUES = [0.5, 0.2, 1.0, 0.3, 0.1, -0.4]
DONES = [False, True, False, False, True]
LAST_VALUE = 0.7


def reference_gae(dones):
    """Advantages and returns from Stable-Baselines3's own rollout buffer."""
    reference = RolloutBuffer(
        len(REWARDS), spaces.Box(0, 1, (1,)), spaces.Discrete(2), device="cpu",
        gamma=GAMMA, gae_lambda=GAE_LAMBDA,
    )
    episode_start = False
    for reward, value, done in zip(REWARDS, VALUES, dones):
        reference.add(
            np.zeros((1, 1)), np.zeros(1), np.array([reward]), np.array([episode_start]),
            th.tensor([value]), th.tensor([0.0]),
        )
        episode_start = done
    reference.compute_returns_and_advantage(th.tensor([LAST_VALUE]), np.array([dones[-1]]))
    return reference.advantages.flatten(), reference.returns.flatten()


@pytest.mark.parametrize("last_done", [False, True])
def test_advantages_match_reference_gae(last_done):
    dones = DONES + [last_done]
    buffer = make_buffer(SINGLE_FEATURE, size=len(REWARDS))
    for reward, value, done in zip(REWARDS, VALUES, dones):
        add_step(buffer, reward=reward, value=value, done=done)
    buffer.compute_returns_and_advantage(last_values=th.tensor([[LAST_VALUE]]), dones=last_done)

    advantages, returns = reference_gae(dones)

    np.testing.assert_allclose(buffer.advantages, advantages, rtol=1e-5)
    np.testing.assert_allclose(buffer.returns, returns, rtol=1e-5)


def epoch_order(buffer, batch_size):
    return np.concatenate([batch["rewards"].numpy() for batch in buffer.get(batch_size)])


def test_minibatches_cover_every_step_once_in_a_new_order_each_epoch():
    buffer = make_buffer(SINGLE_FEATURE, size=30)
    for step in range(30):
        add_step(buffer, reward=float(step))
    buffer.compute_returns_and_advantage(last_values=th.tensor([[0.0]]), dones=False)

    first, second = epoch_order(buffer, 8), epoch_order(buffer, 8)

    np.testing.assert_array_equal(np.sort(first), np.arange(30))
    np.testing.assert_array_equal(np.sort(second), np.arange(30))
    assert not np.array_equal(first, second)


def test_minibatch_observations_keep_their_space_shape(observation_space):
    buffer = make_buffer(observation_space, size=10)
    fill(buffer)

    batches = list(buffer.get(4))

    assert [len(batch["actions"]) for batch in batches] == [4, 4, 2]
    for key, space in observation_space.spaces.items():
        assert batches[-1]["observations"][key].shape == (2, *space.shape), key
