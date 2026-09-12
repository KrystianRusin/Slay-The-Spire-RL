import numpy as np
from stable_baselines3.common.buffers import RolloutBuffer


class CustomRolloutBuffer(RolloutBuffer):
    """Single-environment rollout buffer for dict observations.

    Every per-step array is one-dimensional of buffer length; each observation
    component is stored at its own space shape. The base constructor sizes all
    of them by calling reset.
    """

    def reset(self):
        self.observations = {key: np.zeros((self.buffer_size, *space.shape), dtype=space.dtype)
                             for key, space in self.observation_space.spaces.items()}
        self.actions = np.zeros((self.buffer_size, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size,), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size,), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size,), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size,), dtype=np.float32)
        self.old_log_prob = np.zeros((self.buffer_size,), dtype=np.float32)
        self.values = np.zeros((self.buffer_size,), dtype=np.float32)
        self.pos = 0
        self.full = False

    def add(self, obs, action, reward, done, value, log_prob):
        """Add one transition. done marks whether this transition ended the episode."""
        idx = self.pos

        for key, space in self.observation_space.spaces.items():
            self.observations[key][idx] = obs[key].detach().cpu().numpy().reshape(space.shape)

        self.actions[idx] = action.cpu().numpy()
        self.rewards[idx] = reward
        self.dones[idx] = done
        self.old_log_prob[idx] = log_prob.detach().cpu().item()
        self.values[idx] = value.detach().cpu().item()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size=None):
        """Yield minibatches over the filled steps, in a fresh random order on every call."""
        if batch_size is None:
            batch_size = self.pos
        indices = np.random.permutation(self.pos)

        for start in range(0, self.pos, batch_size):
            batch_indices = indices[start:start + batch_size]
            yield {
                "observations": {key: self.to_torch(obs[batch_indices]) for key, obs in self.observations.items()},
                "actions": self.to_torch(self.actions[batch_indices]),
                "rewards": self.to_torch(self.rewards[batch_indices]),
                "dones": self.to_torch(self.dones[batch_indices]),
                "values": self.to_torch(self.values[batch_indices]),
                "log_probs": self.to_torch(self.old_log_prob[batch_indices]),
                "advantages": self.to_torch(self.advantages[batch_indices]),
                "returns": self.to_torch(self.returns[batch_indices]),
            }

    def compute_returns_and_advantage(self, last_values, dones):
        """GAE(lambda) advantages and TD(lambda) returns over the filled steps.

        last_values is the value of the state after the final step, and dones
        whether that final step ended the episode.
        """
        last_value = last_values.detach().cpu().item()
        last_done = np.asarray(dones, dtype=np.float32).item()

        last_gae_lam = 0.0
        for step in reversed(range(self.pos)):
            if step == self.pos - 1:
                next_non_terminal = 1.0 - last_done
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step]
                next_value = self.values[step + 1]

            delta = self.rewards[step] + self.gamma * next_value * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam

        self.returns = self.advantages + self.values

    def __len__(self):
        return self.pos
