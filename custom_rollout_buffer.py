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
        self.device = device

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
        # Ensure data is processed on the correct device (GPU)
        device = self.device  # Assuming self.device holds the appropriate device (GPU or CPU)

        # Prepare and flatten the data
        for start in range(0, self.pos, batch_size):
            end = start + batch_size

            # Flatten observations and other data
            obs_batch = {key: self.observations[key][start:end] for key in self.observations}

            # Convert to torch tensors and move to GPU
            yield {
                "observations": {key: self.to_torch(obs).view(batch_size, -1).to(device) for key, obs in obs_batch.items()},
                "actions": self.to_torch(self.actions[start:end]).to(device),
                "rewards": self.to_torch(self.rewards[start:end]).to(device),
                "dones": self.to_torch(self.dones[start:end]).to(device),
                "values": self.to_torch(self.values[start:end]).to(device),
                "log_probs": self.to_torch(self.old_log_prob[start:end]).flatten().to(device),
                "advantages": self.to_torch(self.advantages[start:end]).flatten().to(device),
                "returns": self.to_torch(self.returns[start:end]).to(device)
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
