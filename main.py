import json
import socket
import time
import os
import numpy as np
import torch as th
from stable_baselines3.common.buffers import RolloutBuffer
from sb3_contrib.ppo_mask import MaskablePPO, MultiInputPolicy  # Import the MaskableMultiInputPolicy
from custom_rollout_buffer import CustomRolloutBuffer
import matplotlib.pyplot as plt
from collections import deque
from slay_the_spire_env import SlayTheSpireEnv

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

def handle_end_of_episode(client_socket):
    """
    Handles the end-of-episode scenario by sending the "PROCEED" command twice,
    waiting for the game state to update between the two sends.
    """
    # First "PROCEED" command
    proceed_command = "PROCEED"
    client_socket.sendall(proceed_command.encode('utf-8'))
    print("Sent 'PROCEED' command")

    # Wait for the game state update
    try:
        game_state = receive_full_json(client_socket)
        print("Game state received after first 'PROCEED'")
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON after first 'PROCEED': {e}")
        return  # Exit if there's an error

    # Second "PROCEED" command
    client_socket.sendall(proceed_command.encode('utf-8'))
    print("Sent 'PROCEED' command again")

    # Optionally, wait for another game state update if needed
    try:
        game_state = receive_full_json(client_socket)
        print("Game state received after second 'PROCEED'")
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON after second 'PROCEED': {e}")

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
    env = SlayTheSpireEnv({})

    # Initialize MaskablePPO agent with MaskableMultiInputPolicy to handle dict observation spaces
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    if device == th.device("cpu"):
        th.set_num_threads(4)
    else:
        th.cuda.set_per_process_memory_fraction(0.5, device=0)

    model = MaskablePPO("MultiInputPolicy", env, ent_coef=0.03, gamma=0.97, learning_rate=0.0003, clip_range=0.3, verbose=1, device=device)

    # Load model weights if available
    if os.path.exists("maskable_ppo_slay_the_spire.zip"):
        print("Loading existing model weights...")
        model = MaskablePPO.load("maskable_ppo_slay_the_spire", env=env, policy_kwargs={'net_arch': [dict(pi=[64, 64], vf=[64, 64])]})

    # Initialize the rollout buffer
    n_steps = 2048  # Number of steps to collect before updating the model
    total_steps = 100000  # Define the total number of training steps you plan to run

    rollout_buffer = CustomRolloutBuffer(
        buffer_size=n_steps,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=model.device,
        gamma=model.gamma,
        gae_lambda=model.gae_lambda,
        n_envs=1
    )

    current_step = 0  # Track the number of steps completed

    episode = 0
    while True:
        done = False
        total_reward = 0
        episode_length = 0
        obs = env.reset()

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
            env.update_game_state(game_state)

            # Flatten the observation
            obs = env.flatten_observation(game_state)

            obs_tensor = {key: th.tensor(value, dtype=th.float32).unsqueeze(0).to(model.device) for key, value in obs.items()}

            # Get valid action mask
            action_mask = env.get_invalid_action_mask(game_state)
            print(f"Action Mask: {action_mask}")  # Debugging: Print the action mask
            valid_actions = np.where(action_mask)[0]
            print(f"Valid Actions (indices): {valid_actions}")  # Debugging: Print valid action indices
            action_mask_tensor = th.tensor(action_mask, dtype=th.bool).unsqueeze(0).to(model.device)

            # Convert observations and action masks to numpy arrays
            obs_numpy = {key: value.cpu().numpy() for key, value in obs_tensor.items()}
            action_mask_numpy = action_mask_tensor.cpu().numpy()

            action, _states = model.predict(obs_numpy, action_masks=action_mask_numpy)

            # Ensure action is an integer scalar
            action = int(action)  # Convert to an integer if it's a NumPy array or similar
            print(f"Chosen Action: {action}")  # Debugging: Print the chosen action

            if action not in valid_actions:
                print(f"Warning: Chosen action {action} is not in the list of valid actions!")

            chosen_command = env.actions[action]
            print(f"Chosen Action: {action}, Command: {chosen_command}")

            # Send the chosen command to the game
            client_socket.sendall(chosen_command.encode('utf-8'))
        
            # Step through the environment, passing only the action
            new_obs, reward, done, info = env.step(action)
            total_reward += reward
            print("REWARD: ", total_reward)
            episode_length += 1

            new_obs_tensor = {key: th.tensor(value, dtype=th.float32).unsqueeze(0).to(model.device) for key, value in new_obs.items()}

            # Convert action to tensor
            action_tensor = th.tensor(action, dtype=th.long).to(model.device)

            # Predict log_prob and values
            values, log_prob, entropy = model.policy.evaluate_actions(obs_tensor, action_tensor)
            values = model.policy.predict_values(obs_tensor)

            # Store experience in the rollout buffer
            rollout_buffer.add(
                obs_tensor,
                action_tensor,
                reward,
                done,
                values,
                log_prob
            )

            obs = new_obs

            current_step += 1  # Increment the current step count

            # If the buffer is full, update the model
            if len(rollout_buffer) >= n_steps:
                rollout_buffer.compute_returns_and_advantage(last_values=model.policy.predict_values(new_obs_tensor), dones=done)
                update_model(model, rollout_buffer, current_step, total_steps)

                rollout_buffer.reset()

        # Episode ended: update performance metrics
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
        handle_end_of_episode(client_socket)

        # Save model after each episode
        model.save("ppo_slay_the_spire")

    client_socket.close()

def update_model(model, rollout_buffer, current_step, total_steps):
    n_epochs = 10
    batch_size = 64

    progress_remaining = 1 - (current_step / total_steps)

    for epoch in range(n_epochs):
        for rollout_data in rollout_buffer.get(batch_size):
            # Access the actions using the dictionary key
            actions = th.tensor(rollout_data["actions"], dtype=th.long).flatten().to(model.device)

            # Access the observations using the dictionary key
            observations = rollout_data["observations"]  # This will be a dictionary of observation components

            # Convert the observations dictionary back to the format that the model expects (e.g., a tensor or dict of tensors)
            observations_tensor = {key: th.tensor(value).to(model.device) for key, value in observations.items()}

            # Evaluate actions using the model's policy
            values, log_prob, entropy = model.policy.evaluate_actions(observations_tensor, actions)

            # Compute the loss
            advantages = rollout_data["advantages"].to(model.device)
            log_probs_old = rollout_data["log_probs"].to(model.device)
            returns = rollout_data["returns"].to(model.device)

            ratio = th.exp(log_prob - log_probs_old)

            clip_range = model.clip_range(progress_remaining)
            policy_loss_1 = advantages * ratio
            policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
            policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

            # Calculate value loss
            value_loss = th.nn.functional.mse_loss(returns, values)
            print(value_loss)

            entropy_loss = -th.mean(entropy)

            # Combine losses
            loss = policy_loss + model.ent_coef * entropy_loss + model.vf_coef * value_loss

            # Backpropagate the loss and update the model
            model.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(model.policy.parameters(), model.max_grad_norm)
            model.policy.optimizer.step()
            print("Model Updated")

if __name__ == "__main__":
    main()
