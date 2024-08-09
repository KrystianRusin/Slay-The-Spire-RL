from rl.agents.dqn import DQNAgent
import numpy as np

class MaskedDQNAgent(DQNAgent):
    def forward(self, observation):
        state = np.array(observation['state'])
        invalid_action_mask = observation['invalid_action_mask']
        logits = self.model.predict(np.expand_dims(state, axis=0))[0]
        masked_logits = np.where(invalid_action_mask, -1e8, logits)
        action = np.argmax(masked_logits)
        return action