import json
import time
import numpy as np
import socket
from slay_the_spire_env import SlayTheSpireEnv, MaskedSlayTheSpireEnv
from masked_dqn_agent import MaskedDQNAgent
from build_model import build_model
from prepare_state_inputs import prepare_state_inputs

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

    # Load the model
    env = SlayTheSpireEnv({})
    env = MaskedSlayTheSpireEnv(env)
    nb_actions = env.action_space.n
    model = build_model(env.observation_space, nb_actions)
    model.summary()

    done = False
    while not done:
        # Receive the game_state from the middleman process using the new function
        try:
            game_state = receive_full_json(client_socket)
        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON: {e}")
            continue

        # Print the available commands from the game_state
        available_commands = game_state.get('available_commands', [])
        print("Available commands:", available_commands)

        # Update the environment with the new game state
        env.state = game_state

        # Prepare the state inputs for the model
        state_inputs = prepare_state_inputs(env.state)

        # Get the predicted action values from the model
        action_values = model.predict(state_inputs)[0]

        # Get the invalid action mask
        invalid_action_mask = env.get_invalid_action_mask()

        # Apply the invalid action mask to the action values
        masked_action_values = np.where(invalid_action_mask, -1e8, action_values)

        # Choose the action with the highest valid value
        chosen_action = np.argmax(masked_action_values)
        chosen_command = env.actions[chosen_action]
        
        print("Chosen action index:", chosen_action)
        print("Chosen action:", chosen_command)

        # Step in the environment with the chosen action
        state, reward, done, info = env.step(chosen_action, game_state)

        # Send the chosen command back to the middleman process
        client_socket.sendall(chosen_command.encode('utf-8'))

    client_socket.close()

if __name__ == "__main__":
    main()