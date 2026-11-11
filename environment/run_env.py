import copy
import torch as th
import socket
import os
import sys
import time
from collections import deque
from slay_the_spire_env import SlayTheSpireEnv
from model.custom_rollout_buffer import CustomRolloutBuffer
from model.policy import TrainingConfig, make_model
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

def act(model, obs_tensor, action_mask):
    """Sample an action from the masked policy. Returns (action, value, log_prob).

    log_prob is taken from the same masked distribution the action was drawn
    from, which is what the learner recomputes for the PPO ratio.
    """
    mask = th.as_tensor(action_mask, dtype=th.bool, device=model.device).reshape(1, -1)
    with th.no_grad():
        distribution = model.policy.get_distribution(obs_tensor, action_masks=mask)
        action = distribution.get_actions(deterministic=False)
        log_prob = distribution.log_prob(action)
        value = model.policy.predict_values(obs_tensor)
    return int(action.item()), value, log_prob


def run_environment(env_id, port, experience_queue, host="localhost",
                    config=None, initial_state_dict=None, next_weights=None):
    """
    Function to run a single agent in a separate environment.

    config is the TrainingConfig to build the policy from (the learner's, sent
    with its weights); defaults to model/policy.py. initial_state_dict, if
    given, replaces the policy's random initial weights, so the actor plays with
    the learner's network. next_weights, if given, is called after each rollout is
    handed off and returns the state_dict to play the next one with.
    """
    config = config or TrainingConfig()
    n_steps = config.n_steps
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
    model = make_model(config, env=env, device=device)
    if initial_state_dict is not None:
        model.policy.load_state_dict(initial_state_dict)

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
            action, values, log_prob = act(model, obs_tensor, action_mask)
            chosen_command = env.actions[action].text
            connection.send(chosen_command)

            if TRACK_DB:
                game_id = process_game_state(game_state, chosen_command, game_id)

            new_obs, reward, done, info = env.step(action)
            total_reward += reward
            episode_length += 1

            new_obs_tensor = {key: th.tensor(value, dtype=th.float32).unsqueeze(0).to(device) for key, value in new_obs.items()}
            action_tensor = th.tensor(action, dtype=th.long).to(device)

            rollout_buffer.add(
                obs_tensor,
                action_tensor,
                reward,
                done,
                values,
                log_prob,
                action_mask,
            )

            obs = new_obs
            current_step += 1

            if len(rollout_buffer) >= n_steps:
                rollout_buffer.compute_returns_and_advantage(last_values=model.policy.predict_values(new_obs_tensor), dones=done)
                hand_off_rollout(rollout_buffer, experience_queue)
                if next_weights is not None:
                    model.policy.load_state_dict(next_weights())
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
