import torch as th
import socket
import os
from collections import deque
from sb3_contrib.ppo_mask import MaskablePPO
from slay_the_spire_env import SlayTheSpireEnv
from model.custom_rollout_buffer import CustomRolloutBuffer
from communication import receive_full_json, handle_end_of_episode
from plotting import plot_performance_metrics
import json

def run_environment(env_id, port, experience_queue, n_steps=2048):
    """
    Function to run a single agent in a separate environment.
    """
    episode_rewards = []
    episode_lengths = []
    reward_queue = deque(maxlen=10)
    highest_reward = float('-inf')
    
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.settimeout(10)
    client_socket.connect(("localhost", port))

    # Initialize the environment
    env = SlayTheSpireEnv({})
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    model = MaskablePPO("MultiInputPolicy", env, ent_coef=0.03, gamma=0.97, learning_rate=0.0003, clip_range=0.3, verbose=1, device=device)
    
    reload_interval = 100
    reload_counter = 0

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
            try:
                game_state = receive_full_json(client_socket)
            except json.JSONDecodeError as e:
                print(f"Failed to decode JSON in environment {env_id}: {e}")
                continue
            except ConnectionError as e:
                print(f"Connection error in environment {env_id}: {e}")
                break
            print(f"Environment {env_id}: Game State Received")

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

            if len(rollout_buffer) >= n_steps:
                rollout_buffer.compute_returns_and_advantage(last_values=model.policy.predict_values(new_obs_tensor), dones=done)
                experience_queue.put(rollout_buffer)
                rollout_buffer.reset()

                reload_counter += 1
                if reload_counter % reload_interval == 0:
                    if os.path.exists("maskable_ppo_slay_the_spire.zip"):
                        model = MaskablePPO.load("maskable_ppo_slay_the_spire", env=env)
                        print(f"Environment {env_id}: Reloaded updated model weights.")

        episode_rewards.append(total_reward)
        episode_lengths.append(episode_length)
        reward_queue.append(total_reward)
        highest_reward = max(highest_reward, total_reward)
        rolling_avg = sum(reward_queue) / len(reward_queue) if reward_queue else 0

        if episode % 10 == 0:
            plot_performance_metrics(episode_rewards, episode_lengths, [rolling_avg], highest_reward)

        episode += 1
        handle_end_of_episode(client_socket)

    client_socket.close()