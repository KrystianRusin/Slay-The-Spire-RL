import json
import socket
import time
import os
import numpy as np
import torch
import stable_baselines3 as sb3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.ppo import PPO
from slay_the_spire_env import SlayTheSpireEnv, MaskedSlayTheSpireEnv
import matplotlib.pyplot as plt


def plot_performance_metrics(episode_rewards, episode_lengths, save_path="performance_metrics.png"):
    # Create the figure and axis objects for the plot
    fig, ax1 = plt.subplots()

    # Plot the total reward per episode
    ax1.plot(range(len(episode_rewards)), episode_rewards, 'b-', label='Reward')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward', color='b')
    ax1.tick_params('y', colors='b')

    # Create another y-axis sharing the same x-axis for episode length
    ax2 = ax1.twinx()
    ax2.plot(range(len(episode_lengths)), episode_lengths, 'r-', label='Episode Length')
    ax2.set_ylabel('Episode Length', color='r')
    ax2.tick_params('y', colors='r')

    # Add a title and grid
    plt.title('Performance Metrics: Rewards and Episode Lengths')
    plt.grid(True)

    # Save the plot to the specified file path (overwrites if file exists)
    plt.savefig(save_path, format='png')

    # Close the plot to free up memory
    plt.close(fig)

    # Notify that the plot has been saved
    print(f"Performance metrics saved to {save_path}")


def receive_full_json(client_socket):
    data = b''
    while True:
        part = client_socket.recv(4096)
        data += part
        try:
            return json.loads(data.decode('utf-8'))
        except json.JSONDecodeError:
            if not part:
                raise


def main():
    episode_rewards = []
    episode_lengths = []

    # Establish socket connection to receive the game state
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(("localhost", 9999))

    # Initialize the environment
    unmasked_env = SlayTheSpireEnv({})
    env = MaskedSlayTheSpireEnv(unmasked_env)

    # Initialize PPO agent with MultiInputPolicy to handle dict observation spaces
    model = PPO("MultiInputPolicy", env, ent_coef=0.03,  gamma=0.97, learning_rate=0.0003, clip_range=0.3, verbose=1)

    # Load model weights if available
    if os.path.exists("ppo_slay_the_spire.zip"):
        print("Loading existing model weights...")
        model = PPO.load("ppo_slay_the_spire", env=env)

    episode = 0
    while True:
        done = False
        total_reward = 0
        episode_length = 0

        while not done:
            # Receive the next game state via socket
            time.sleep(1)  # Ensure we are not overwhelming the socket
            try:
                game_state = receive_full_json(client_socket)
            except json.JSONDecodeError as e:
                print(f"Failed to decode JSON: {e}")
                continue
            print("Game State Received")

            # Update the environment's internal state with the new game state
            unmasked_env.update_game_state(game_state)

            # Ensure the invalid action mask is recalculated with the new state
            invalid_action_mask = env.get_invalid_action_mask()

            # Debug: Print available commands and valid actions after the mask is applied
            available_commands = game_state.get('available_commands', [])
            print("Available Commands:", available_commands)

            # Print the action space with validity (for debugging purposes)
            env.print_action_space_with_validity()

            # Predict the next action using the PPO model
            observation = env.flatten_observation(game_state)

            # Predict the action using the model
            action_logits, _states = model.predict(observation)

            # Apply the invalid action mask to the action logits
            masked_logits = np.where(invalid_action_mask, -1e8, action_logits)

                   
            # Choose the action with the highest valid value
            chosen_action = np.argmax(masked_logits)
            chosen_command = env.actions[chosen_action]
            print(f"Chosen Action: {chosen_action}, Command: {chosen_command}")

            # Send the chosen command to the game
            client_socket.sendall(chosen_command.encode('utf-8'))

            # Step through the environment, passing only the action
            observation, reward, done, info = env.step(chosen_action)

            # Accumulate reward and step count
            total_reward += reward
            print("TOTAL REWARD", total_reward)
            episode_length += 1
            print("\n")

          # Append metrics for plotting
        episode_rewards.append(total_reward)
        episode_lengths.append(episode_length)

        # Periodically plot the metrics
        if episode % 10 == 0:
            model.save("ppo_slay_the_spire")
            

        # Increment episode count
        episode += 1

        # Save model after each episode
        plot_performance_metrics(episode_rewards, episode_lengths)
        
    client_socket.close()



if __name__ == "__main__":
    main()
