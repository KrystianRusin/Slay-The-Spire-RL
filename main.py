import json
import socket
import time
import os
import numpy as np
import torch as th
from sb3_contrib.ppo_mask import MaskablePPO
from custom_rollout_buffer import CustomRolloutBuffer
import matplotlib.pyplot as plt
from collections import deque
from slay_the_spire_env import SlayTheSpireEnv
from multiprocessing import Process, Queue

def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.
    """
    def _init():
        env = SlayTheSpireEnv({})
        env.seed(seed + rank)
        return env
    return _init

# Number of parallel environments
n_envs = 4  # Adjust this number based on your system's capability

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
    Handles the end-of-episode scenario by sending the necessary commands
    to navigate through the game over screen and start a new game.
    """
    # Sequence of commands to navigate back to the main menu and start a new game
    commands = [
        "PROCEED",     # To proceed from the game over screen
        "PROCEED",     # To confirm and return to main menu
        "START_GAME",  # Command to start a new game
        # Add any additional commands required to start a new run
    ]

    for command in commands:
        client_socket.sendall(command.encode('utf-8'))
        print(f"Sent '{command}' command")

        # Wait for the game state update
        try:
            game_state = receive_full_json(client_socket)
            print(f"Game state received after '{command}'")
        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON after '{command}': {e}")
            return  # Exit if there's an error
        except ConnectionError as e:
            print(f"Connection error after '{command}': {e}")
            return  # Handle connection errors

def receive_full_json(client_socket):
    data = b''
    while True:
        try:
            part = client_socket.recv(4096)
            if not part:
                # No data received, connection might be closed
                raise ConnectionError("Socket connection closed")
            data += part
            try:
                return json.loads(data.decode('utf-8'))
            except json.JSONDecodeError:
                # Incomplete data, continue receiving
                continue
        except socket.timeout:
            # Timeout occurred, handle accordingly
            print("Timeout occurred while receiving game state. Requesting resend...")
            # Optionally, send a message back to middleman to resend the game state
            continue  # Continue trying to receive data

def run_environment(env_id, port, experience_queue):
    """
    Function to run a single agent in a separate environment.
    """
    episode_rewards = []
    episode_lengths = []
    reward_queue = deque(maxlen=10)
    highest_reward = float('-inf')
    
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.settimeout(10)  # Set a timeout
    client_socket.connect(("localhost", port))

    # Initialize the environment
    env = SlayTheSpireEnv({})
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    model = MaskablePPO("MultiInputPolicy", env, ent_coef=0.03, gamma=0.97, learning_rate=0.0003, clip_range=0.3, verbose=1, device=device)
    
    if os.path.exists("maskable_ppo_slay_the_spire.zip"):
        print(f"Environment {env_id}: Loading existing model weights...")
        model = MaskablePPO.load("maskable_ppo_slay_the_spire", env=env)
    
    n_steps = 2048
    rollout_buffer = CustomRolloutBuffer(
        buffer_size=n_steps,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=model.device,
        gamma=model.gamma,
        gae_lambda=model.gae_lambda,
        n_envs=1
    )

    current_step = 0
    episode = 0
    while True:
        done = False
        total_reward = 0
        episode_length = 0
        obs = env.reset()

        while not done:
            # Remove or reduce the delay
            # time.sleep(1)

            # Receive the next game state via socket
            try:
                game_state = receive_full_json(client_socket)
            except json.JSONDecodeError as e:
                print(f"Failed to decode JSON in environment {env_id}: {e}")
                continue
            except ConnectionError as e:
                print(f"Connection error in environment {env_id}: {e}")
                break  # Exit the loop if the connection is lost
            print(f"Environment {env_id}: Game State Received")

            # Update the environment's internal state with the new game state
            env.update_game_state(game_state)
            obs = env.flatten_observation(game_state)
            obs_tensor = {key: th.tensor(value, dtype=th.float32).unsqueeze(0).to(device) for key, value in obs.items()}

            action_mask = env.get_invalid_action_mask(game_state)
            action_mask_tensor = th.tensor(action_mask, dtype=th.bool).unsqueeze(0).to(device)
            obs_numpy = {key: value.cpu().numpy() for key, value in obs_tensor.items()}
            action_mask_numpy = action_mask_tensor.cpu().numpy()

            action, _states = model.predict(obs_numpy, action_masks=action_mask_numpy)
            action = int(action)
            chosen_command = env.actions[action]
            client_socket.sendall(chosen_command.encode('utf-8'))

            new_obs, reward, done, info = env.step(action)
            total_reward += reward
            episode_length += 1

            new_obs_tensor = {key: th.tensor(value, dtype=th.float32).unsqueeze(0).to(device) for key, value in new_obs.items()}
            action_tensor = th.tensor(action, dtype=th.long).to(device)
            values, log_prob, entropy = model.policy.evaluate_actions(obs_tensor, action_tensor)
            values = model.policy.predict_values(obs_tensor)

            rollout_buffer.add(
                obs_tensor,
                action_tensor,
                reward,
                done,
                values,
                log_prob
            )

            obs = new_obs
            current_step += 1

            # If the buffer is full, send experiences back to the main process
            if len(rollout_buffer) >= n_steps:
                rollout_buffer.compute_returns_and_advantage(last_values=model.policy.predict_values(new_obs_tensor), dones=done)
                experience_queue.put(rollout_buffer)  # Send the filled buffer to the main process
                rollout_buffer.reset()

        episode_rewards.append(total_reward)
        episode_lengths.append(episode_length)
        reward_queue.append(total_reward)
        highest_reward = max(highest_reward, total_reward)
        rolling_avg = sum(reward_queue) / len(reward_queue) if reward_queue else 0

        if episode % 10 == 0:
            plot_performance_metrics(episode_rewards, episode_lengths, [rolling_avg], highest_reward)

        episode += 1
        handle_end_of_episode(client_socket)
        model.save(f"maskable_ppo_slay_the_spire")

    client_socket.close()

def main():
    num_envs = 4  # Adjust as needed
    base_port = 9999
    experience_queue = Queue()
    processes = []

    for env_id in range(num_envs):
        port = base_port + env_id
        p = Process(target=run_environment, args=(env_id, port, experience_queue))
        p.start()
        processes.append(p)

    model = MaskablePPO("MultiInputPolicy", SlayTheSpireEnv({}), ent_coef=0.03, gamma=0.97, learning_rate=0.0003, clip_range=0.3, verbose=1, device=th.device("cuda" if th.cuda.is_available() else "cpu"))
    
    n_steps = 2048
    total_steps = 100000
    current_step = 0
    
    while current_step < total_steps:
        # Collect experiences from all environment processes
        experiences = []
        for _ in range(num_envs):
            experiences.append(experience_queue.get())

        # Aggregate experiences and update the model
        for exp in experiences:
            update_model(model, exp, current_step, total_steps)

        current_step += n_steps * num_envs  # Increment step count by the total number of steps processed

    for p in processes:
        p.join()

def update_model(model, rollout_buffer, current_step, total_steps):
    n_epochs = 10
    batch_size = 64
    progress_remaining = 1 - (current_step / total_steps)

    for epoch in range(n_epochs):
        for rollout_data in rollout_buffer.get(batch_size):
            actions = th.tensor(rollout_data["actions"], dtype=th.long).flatten().to(model.device)
            observations = rollout_data["observations"]
            observations_tensor = {key: th.tensor(value).to(model.device) for key, value in observations.items()}
            values, log_prob, entropy = model.policy.evaluate_actions(observations_tensor, actions)

            advantages = rollout_data["advantages"].to(model.device)
            log_probs_old = rollout_data["log_probs"].to(model.device)
            returns = rollout_data["returns"].to(model.device)
            ratio = th.exp(log_prob - log_probs_old)
            clip_range = model.clip_range(progress_remaining)
            policy_loss_1 = advantages * ratio
            policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
            policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
            value_loss = th.nn.functional.mse_loss(returns, values)
            entropy_loss = -th.mean(entropy)
            loss = policy_loss + model.ent_coef * entropy_loss + model.vf_coef * value_loss
            model.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(model.policy.parameters(), model.max_grad_norm)
            model.policy.optimizer.step()

            # Logging the model update with timestamp
            with open("model_update_log.txt", "a") as log_file:
                log_file.write(f"Model was updated at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            print("Model Updated and logged.")

if __name__ == "__main__":
    main()
