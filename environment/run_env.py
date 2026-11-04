import copy
import torch as th
import socket
import os
import sys
import time
from collections import deque
from sb3_contrib.ppo_mask import MaskablePPO
from slay_the_spire_env import SlayTheSpireEnv
from model.custom_rollout_buffer import CustomRolloutBuffer
from util.communication import FramedConnection, handle_end_of_episode
from util.plotting import plot_performance_metrics
from util.data_processor import process_game_state
import json
from util.game_over_tracking import update_game_stats_on_game_over
from util.card_tracking import track_card_performance

# Set to 0 to skip the Postgres stat tracking and the performance plot, e.g. in a
# container with no database. Both default to on, as before.
TRACK_DB = os.environ.get("STS_TRACK_DB", "1") != "0"
PLOT = os.environ.get("STS_PLOT", "1") != "0"


def connect_with_retry(host, port, timeout=180):
    """Connect to the middleman, waiting for it to come up.

    The middleman only exists once the game has booted and CommunicationMod has
    started it, which takes a while, so a single attempt is not enough.
    """
    deadline = time.monotonic() + timeout
    while True:
        try:
            return socket.create_connection((host, port), timeout=10)
        except OSError as e:
            if time.monotonic() > deadline:
                raise ConnectionError(f"Middleman at {host}:{port} not reachable after {timeout}s") from e
            time.sleep(2)


def hand_off_rollout(rollout_buffer, experience_queue):
    """Queue a completed rollout for the learner and clear the buffer for the next one."""
    # Queue.put pickles on a background thread, so it must get a copy the reset cannot reach.
    experience_queue.put(copy.deepcopy(rollout_buffer))
    rollout_buffer.reset()

def run_environment(env_id, port, experience_queue, n_steps=2048, host="localhost"):
    """
    Function to run a single agent in a separate environment.
    """
    episode_rewards = []
    episode_lengths = []
    reward_queue = deque(maxlen=10)
    highest_reward = float('-inf')
    
    client_socket = connect_with_retry(host, port)
    # Block as long as the game needs: the middleman resolves stalls itself, and
    # a dead middleman closes the socket, which raises ConnectionError below.
    client_socket.settimeout(None)
    connection = FramedConnection(client_socket)
    print(f"Environment {env_id}: connected to middleman at {host}:{port}", flush=True)

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
        game_id = None

        while not done:
            try:
                game_state = connection.receive_json()
            except json.JSONDecodeError as e:
                print(f"Failed to decode JSON in environment {env_id}: {e}")
                continue
            except ConnectionError as e:
                # The game or middleman is gone; there is nothing to recover here.
                print(f"Connection error in environment {env_id}: {e}", flush=True)
                client_socket.close()
                sys.exit(1)

            env.update_game_state(game_state)
            obs = env.flatten_observation(game_state)
            obs_tensor = {key: th.tensor(value, dtype=th.float32).unsqueeze(0).to(device) for key, value in obs.items()}

            action_mask = env.get_valid_action_mask(game_state)
            action_mask_tensor = th.tensor(action_mask, dtype=th.bool).unsqueeze(0).to(device)
            obs_numpy = {key: value.cpu().numpy() for key, value in obs_tensor.items()}
            action_mask_numpy = action_mask_tensor.cpu().numpy()

            action, _states = model.predict(obs_numpy, action_masks=action_mask_numpy)
            action = int(action.item())  # predict returns shape (1,) for a batch of one
            chosen_command = env.actions[action].text
            connection.send(chosen_command)

            if TRACK_DB:
                game_id = process_game_state(game_state, chosen_command, game_id)

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
                hand_off_rollout(rollout_buffer, experience_queue)

                reload_counter += 1
                if reload_counter % reload_interval == 0:
                    if os.path.exists("maskable_ppo_slay_the_spire.zip"):
                        model = MaskablePPO.load("maskable_ppo_slay_the_spire", env=env)
                        print(f"Environment {env_id}: Reloaded updated model weights.")
            if done:
                screen_state = game_state['game_state'].get('screen_state', {})
                victory = screen_state.get('victory', False)
                floor_reached = game_state['game_state'].get('floor', 0)

                character = game_state['game_state'].get('class', '?')
                print(f"Environment {env_id}: episode {episode} done: {character} "
                      f"floor={floor_reached} victory={victory} reward={total_reward:.1f} "
                      f"steps={episode_length}", flush=True)

                # Update the game stats and card performance before sending any commands
                if TRACK_DB:
                    update_game_stats_on_game_over(game_state, game_id, total_reward)
                    track_card_performance(game_state['game_state'], floor_reached, victory)

        episode_rewards.append(total_reward)
        episode_lengths.append(episode_length)
        reward_queue.append(total_reward)
        highest_reward = max(highest_reward, total_reward)
        rolling_avg = sum(reward_queue) / len(reward_queue) if reward_queue else 0

        if PLOT and episode % 10 == 0:
            plot_performance_metrics(episode_rewards, episode_lengths, [rolling_avg], highest_reward)

        episode += 1
  
        handle_end_of_episode(connection)

    client_socket.close()
