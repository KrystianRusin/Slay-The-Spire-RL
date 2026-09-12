"""Build rollout buffers filled the way the actor loop fills them."""

import torch as th
from gymnasium import spaces

from model.custom_rollout_buffer import CustomRolloutBuffer

GAMMA = 0.97
GAE_LAMBDA = 0.95


def make_buffer(observation_space, size=8):
    return CustomRolloutBuffer(
        buffer_size=size,
        observation_space=observation_space,
        action_space=spaces.Discrete(141),
        device="cpu",
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        n_envs=1,
    )


def add_step(buffer, reward=0.0, done=False, value=0.0, log_prob=0.0, action=0):
    """Add one transition with the tensor shapes run_env produces."""
    obs = {
        key: th.rand(1, *space.shape)
        for key, space in buffer.observation_space.spaces.items()
    }
    buffer.add(
        obs,
        th.tensor(action, dtype=th.long),
        reward,
        done,
        th.tensor([[value]]),
        th.tensor([log_prob]),
    )


def fill(buffer):
    """Fill every step with varied transitions, then compute returns."""
    generator = th.Generator().manual_seed(0)
    for step in range(buffer.buffer_size):
        add_step(
            buffer,
            reward=float(th.randn(1, generator=generator)),
            value=float(th.randn(1, generator=generator)),
            log_prob=-float(th.rand(1, generator=generator)) * 3,
            action=step % 5,
            done=step % 7 == 6,
        )
    buffer.compute_returns_and_advantage(last_values=th.tensor([[0.0]]), dones=False)
    return buffer
