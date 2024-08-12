import numpy as np
from tensorflow import keras
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy

class MaskedDQNAgent(DQNAgent):
    def __init__(self, model, nb_actions, memory, policy, **kwargs):
        super(MaskedDQNAgent, self).__init__(model=model, nb_actions=nb_actions, memory=memory, policy=policy, **kwargs)
        self.target_model = None  # This will be set later

    def compile(self, optimizer, metrics=[]):
        super(MaskedDQNAgent, self).compile(optimizer=optimizer, metrics=metrics)
        # Create a target model
        self.target_model = keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())

    def update_target_model(self):
        # Update the target model weights
        self.target_model.set_weights(self.model.get_weights())

    def train_step(self, state, action, reward, next_state, done):
        target = self.model.predict(state)
        next_q_values = self.target_model.predict(next_state)
        max_next_q_value = np.max(next_q_values, axis=1)
        target[0][action] = reward + (1 - done) * 0.99 * max_next_q_value

        # Train the model with the updated target
        self.model.fit(state, target, verbose=0)

    def forward(self, observation):
        # Extract individual inputs from observation
        state_inputs = observation['state']  # This should be a list of arrays
        
        # Continue with the forward pass using the model with multiple inputs
        invalid_action_mask = observation['invalid_action_mask']
        logits = self.model.predict(state_inputs)

        # Apply the invalid action mask to the logits
        masked_logits = np.where(invalid_action_mask, -1e8, logits)

        # Choose the action with the highest valid value
        action = np.argmax(masked_logits)

        return action