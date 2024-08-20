import numpy as np
from stable_baselines3.common.buffers import RolloutBuffer
from collections import defaultdict

class CustomRolloutBuffer(RolloutBuffer):
    def __init__(self, buffer_size, observation_space, action_space, device, gamma=0.99, gae_lambda=0.95, n_envs=1):
        super().__init__(buffer_size, observation_space, action_space, device, gamma, gae_lambda, n_envs)
        self.observations = {key: np.zeros((self.buffer_size, self.n_envs, *space.shape), dtype=space.dtype)
                             for key, space in observation_space.spaces.items()}

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
                "log_probs": self.log_probs[start:end],
                "advantages": self.advantages[start:end],
            }

    # Method to compute advantages and returns
    def compute_returns_and_advantage(self, last_values, dones):
        # GAE and discounted returns calculation
        for step in reversed(range(self.pos)):
            if step == self.pos - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_values = self.values[step + 1]

            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            self.advantages[step] = delta + self.gamma * self.gae_lambda * next_non_terminal * self.advantages[step + 1]

        self.returns = self.advantages + self.values

    def __len__(self):
        return self.pos
