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
from collections import deque

# Function to plot performance metrics with separate subplots for rewards, rolling averages, and episode lengths
def plot_performance_metrics(episode_rewards, episode_lengths, rolling_avg_rewards, highest_reward, save_path="performance_metrics.png"):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

    # Plot total rewards on the first subplot
    ax1.plot(range(len(episode_rewards)), episode_rewards, 'b-', label='Reward')
    ax1.axhline(y=highest_reward, color='r', linestyle='--', label=f'Highest Reward: {highest_reward}')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Episode Rewards')
    ax1.legend()
    ax1.grid(True)

    # Plot rolling average rewards on the second subplot
    ax2.plot(range(len(rolling_avg_rewards)), rolling_avg_rewards, 'g--', label='Rolling Avg Reward (Last 10)')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Rolling Avg Reward')
    ax2.set_title('Rolling Average of Rewards')
    ax2.legend()
    ax2.grid(True)

    # Plot episode lengths on the third subplot
    ax3.plot(range(len(episode_lengths)), episode_lengths, 'r-', label='Episode Length')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Episode Length')
    ax3.set_title('Episode Lengths')
    ax3.grid(True)

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(save_path, format='png')
    plt.close(fig)

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
    rolling_avg_rewards = []
    reward_queue = deque(maxlen=10)  # Rolling window for the last 10 rewards
    highest_reward = float('-inf')  # Initialize highest reward as negative infinity

    # Establish socket connection to receive the game state
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(("localhost", 9999))

    # Initialize the environment
    unmasked_env = SlayTheSpireEnv({})
    env = MaskedSlayTheSpireEnv(unmasked_env)

    # Initialize PPO agent with MultiInputPolicy to handle dict observation spaces
    model = PPO("MultiInputPolicy", env, ent_coef=0.03, gamma=0.97, learning_rate=0.0003, clip_range=0.3, verbose=1)

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

        # Update performance metrics at the end of the episode
        episode_rewards.append(total_reward)
        episode_lengths.append(episode_length)
        reward_queue.append(total_reward)

        # Update the highest reward if the current total_reward exceeds it
        if total_reward > highest_reward:
            highest_reward = total_reward

        # Calculate the rolling average of the last 10 rewards and append to the rolling average list
        rolling_avg = sum(reward_queue) / len(reward_queue) if reward_queue else 0
        rolling_avg_rewards.append(rolling_avg)

        # Periodically plot the metrics every 10 episodes
        if episode % 10 == 0:
            plot_performance_metrics(episode_rewards, episode_lengths, rolling_avg_rewards, highest_reward)

        # Increment episode count
        episode += 1

        # Save model after each episode
        model.save("ppo_slay_the_spire")

    client_socket.close()


if __name__ == "__main__":
    main()
