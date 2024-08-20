import numpy as np
from stable_baselines3.common.buffers import RolloutBuffer
from collections import defaultdict
import torch as th

class CustomRolloutBuffer(RolloutBuffer):
    def __init__(self, buffer_size, observation_space, action_space, device, gamma=0.99, gae_lambda=0.95, n_envs=1):
        super().__init__(buffer_size, observation_space, action_space, device, gamma, gae_lambda, n_envs)
        self.observations = {key: np.zeros((self.buffer_size, self.n_envs, *space.shape), dtype=space.dtype)
                             for key, space in observation_space.spaces.items()}
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

    def reset(self):
        """
        Reset the buffer by filling it with zeros for each observation component.
        """
        self.observations = {key: np.zeros((self.buffer_size, self.n_envs, *space.shape), dtype=space.dtype)
                             for key, space in self.observation_space.spaces.items()}
        self.actions = np.zeros((self.buffer_size, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size,), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size,), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size,), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size,), dtype=np.float32)
        self.old_log_prob = np.zeros((self.buffer_size,), dtype=np.float32)
        self.log_prob = np.zeros((self.buffer_size,), dtype=np.float32)
        self.values = np.zeros((self.buffer_size,), dtype=np.float32)  # Initialize values here
        self.pos = 0
        self.full = False

    def add(self, obs, action, reward, done, value, log_prob):
        """
        Add a new transition to the buffer.
        """
        idx = self.pos

        # Store dict observation components
        for key in obs:
            self.observations[key][idx] = obs[key].detach().cpu().numpy()

        # Store other values
        self.actions[idx] = action.cpu().numpy()
        self.rewards[idx] = reward
        self.dones[idx] = done
        self.returns[idx] = value.detach().cpu().numpy()
        self.old_log_prob[idx] = log_prob.detach().cpu().numpy()
        self.values[idx] = value.detach().cpu().numpy()  # Store value predictions

        # Update position
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size=None):
        # Return the data in batches, handling the dictionary observation space
        for start in range(0, self.pos, batch_size):
            end = start + batch_size

            # Yield batch of observations as dictionaries
            obs_batch = {key: self.observations[key][start:end] for key in self.observations}

            yield {
                "observations": obs_batch,
                "actions": self.actions[start:end],
                "rewards": self.rewards[start:end],
                "dones": self.dones[start:end],
                "values": self.values[start:end],
                "log_probs": self.old_log_prob[start:end],
                "advantages": self.advantages[start:end],
                "returns": self.returns[start:end]
            }

    # Method to compute advantages and returns
    def compute_returns_and_advantage(self, last_values, dones):
        # GAE and discounted returns calculation
        last_values = last_values.cpu().detach().numpy()  # Convert the last values to numpy array upfront

        for step in reversed(range(self.pos)):
            if step == self.pos - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_values = self.values[step + 1]

            # Ensure next_values is a NumPy array (if not already)
            if isinstance(next_values, th.Tensor):
                next_values = next_values.cpu().detach().numpy()

            # Calculate the delta (TD error) and advantages
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            self.advantages[step] = delta + self.gamma * self.gae_lambda * next_non_terminal * (self.advantages[step + 1] if step < self.pos - 1 else 0)

        # Calculate the returns
        self.returns = self.advantages + self.values

    def __len__(self):
        return self.pos
