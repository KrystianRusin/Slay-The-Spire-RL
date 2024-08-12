import json
import numpy as np
import socket
import time
from slay_the_spire_env import SlayTheSpireEnv, MaskedSlayTheSpireEnv
from masked_dqn_agent import MaskedDQNAgent
from build_model import build_model
from prepare_state_inputs import prepare_state_inputs
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy
from keras.optimizers.legacy import Adam

def receive_full_json(client_socket):
    data = b''
    while True:
        part = client_socket.recv(4096)
        data += part
        try:
            return json.loads(data.decode('utf-8'))
        except json.JSONDecodeError:
            # If a JSONDecodeError occurs, continue receiving more data
            if not part:
                raise  # If no more data is received, re-raise the exception

def main():
    # Connect to the middleman process
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(("localhost", 9999))

    # Load the environment and model
    env = SlayTheSpireEnv({})
    env = MaskedSlayTheSpireEnv(env)
    nb_actions = env.action_space.n
    model = build_model(env.observation_space, nb_actions)
    model.summary()

    # Setup the DQN agent
    memory = SequentialMemory(limit=50000, window_length=1)
    policy = EpsGreedyQPolicy(eps=1.0)  # Start with full exploration
    agent = MaskedDQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=1000, target_model_update=1e-2, policy=policy)
    agent.compile(Adam(learning_rate=1e-3), metrics=['mae'])

    previous_game_state = None  # Track the previous game state

    done = False
    while not done:
        # Receive the game_state from the middleman process
        try:
            game_state = receive_full_json(client_socket)
        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON: {e}")
            continue

        # Check if the game state has changed
        if game_state == previous_game_state:
            print("Game state has not changed. Waiting for the next update...")
            time.sleep(0.1)  # Small delay to avoid tight loop
            continue

        print("New game state received.")
        previous_game_state = game_state  # Update the previous game state

        # Prepare the state inputs for the model
        state_inputs = prepare_state_inputs(env.state)

        # Use the DQN agent to select an action
        action = agent.forward({'state': state_inputs, 'invalid_action_mask': env.get_invalid_action_mask()})
        chosen_command = env.actions[action]

        print("Chosen action index:", action)
        print("Chosen action:", chosen_command)

        # Step in the environment with the chosen action
        state, reward, done, info = env.step(action, game_state)
        # Send the chosen command back to the middleman process
        client_socket.sendall(chosen_command.encode('utf-8'))

        # Store the experience and train the agent
        agent.backward(reward, terminal=done)

    client_socket.close()

if __name__ == "__main__":
    main()