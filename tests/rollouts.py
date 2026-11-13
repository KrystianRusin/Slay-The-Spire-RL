"""Build rollout buffers filled the way the actor loop fills them."""

import numpy as np
import torch as th
from gymnasium import spaces

from environment.run_env import act
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


def mask_for(buffer, action, legal=(0, 1, 2, 3, 4)):
    """A mask with the actions in legal, plus action itself, allowed."""
    mask = np.zeros(buffer.action_space.n, dtype=bool)
    mask[list(legal)] = True
    mask[action] = True
    return mask


def add_step(buffer, reward=0.0, done=False, value=0.0, log_prob=0.0, action=0, action_mask=None):
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
        mask_for(buffer, action) if action_mask is None else action_mask,
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


def play(model, size=128):
    """A rollout played by model the way run_environment plays one."""
    buffer = make_buffer(model.observation_space, size=size)
    generator = th.Generator().manual_seed(2)
    for step in range(size):
        obs = {key: th.rand(1, *space.shape, generator=generator)
               for key, space in model.observation_space.spaces.items()}
        mask = mask_for(buffer, (step * 11) % 141)
        action, value, log_prob = act(model, obs, mask)
        buffer.add(obs, th.tensor(action), float(step % 3), step % 20 == 19, value, log_prob, mask)
    buffer.compute_returns_and_advantage(last_values=th.tensor([[0.0]]), dones=False)
    return buffer
