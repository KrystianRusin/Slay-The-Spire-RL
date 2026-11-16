import copy
import random
import torch as th
import socket
import os
import sys
import time
from collections import deque
from slay_the_spire_env import SlayTheSpireEnv
from model.custom_rollout_buffer import CustomRolloutBuffer
from model.policy import TrainingConfig, make_model
from util.communication import FramedConnection
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


def to_tensors(obs, device):
    return {key: th.tensor(value, dtype=th.float32).unsqueeze(0).to(device) for key, value in obs.items()}


class Actor:
    """Turns each game state into the command to send back, and records PPO steps.

    An action is only finished once the game answers it: the next state says
    what reward it earned and whether it ended the run. So the step taken is
    kept as pending, and goes into the buffer when the following state arrives.

    The game-over screen and the main menu aren't decisions, so they aren't
    recorded: the actor sends PROCEED, then starts a run as a random character.

    on_rollout_full() is called each time the buffer fills, after the returns
    are computed and before the next action is chosen, so it can hand the
    rollout off and swap in new weights.
    """

    def __init__(self, env, model, buffer, on_rollout_full, rng=None, reward_scale=1.0):
        self.env = env
        self.reward_scale = reward_scale
        self.model = model
        self.buffer = buffer
        self.on_rollout_full = on_rollout_full
        self.rng = rng or random.Random()
        self.pending = None  # (obs_tensor, action, value, log_prob, mask) sent, awaiting its result
        self.episode_reward = 0.0
        self.episode_steps = 0
        self.ended = None  # (reward, steps) of the run that ended on the last state, else None

    def on_state(self, state):
        env = self.env
        env.update_game_state(state)
        done = env.check_if_done()
        obs_tensor = to_tensors(env.flatten_observation(state), self.model.device)
        self.ended = None

        if self.pending is not None:
            self._finish_pending(done, obs_tensor)

        if done:
            self.ended = (self.episode_reward, self.episode_steps)
            self.episode_reward, self.episode_steps = 0.0, 0
            return "PROCEED"
        if not state.get("game_state"):
            # Main menu: no run to observe yet. Both characters stay in the mix.
            return self.rng.choice([env.actions[i].text for i in env.actions_by_command["start"]])

        action_mask = env.get_valid_action_mask(state)
        action, value, log_prob = act(self.model, obs_tensor, action_mask)
        env.record_action(action)
        self.pending = (obs_tensor, action, value, log_prob, action_mask)
        return env.actions[action].text

    def _finish_pending(self, done, next_obs_tensor):
        obs_tensor, action, value, log_prob, action_mask = self.pending
        self.pending = None
        reward = self.env.calculate_reward(action)
        self.episode_reward += reward  # unscaled, for the logs
        self.episode_steps += 1
        self.buffer.add(obs_tensor, th.tensor(action, dtype=th.long), reward * self.reward_scale, done,
                        value, log_prob, action_mask)
        if len(self.buffer) >= self.buffer.buffer_size:
            with th.no_grad():
                last_value = self.model.policy.predict_values(next_obs_tensor)
            self.buffer.compute_returns_and_advantage(last_values=last_value, dones=done)
            self.on_rollout_full()


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

    def on_rollout_full():
        hand_off_rollout(rollout_buffer, experience_queue)
        if next_weights is not None:
            model.policy.load_state_dict(next_weights())

    actor = Actor(env, model, rollout_buffer, on_rollout_full, reward_scale=config.reward_scale)
    episode = 0
    game_id = None

    while True:
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

        command = actor.on_state(game_state)
        connection.send(command)

        if TRACK_DB:
            game_id = process_game_state(game_state, command, game_id)

        if actor.ended is not None:
            total_reward, episode_length = actor.ended
            screen_state = game_state['game_state'].get('screen_state', {})
            victory = screen_state.get('victory', False)
            floor_reached = game_state['game_state'].get('floor', 0)
            character = game_state['game_state'].get('class', '?')
            print(f"Environment {env_id}: episode {episode} done: {character} "
                  f"floor={floor_reached} victory={victory} reward={total_reward:.1f} "
                  f"steps={episode_length}", flush=True)

            if TRACK_DB:
                update_game_stats_on_game_over(game_state, game_id, total_reward)
                track_card_performance(game_state['game_state'], floor_reached, victory)
            game_id = None

            episode_rewards.append(total_reward)
            episode_lengths.append(episode_length)
            reward_queue.append(total_reward)
            highest_reward = max(highest_reward, total_reward)
            rolling_avg = sum(reward_queue) / len(reward_queue) if reward_queue else 0
            if PLOT and episode % 10 == 0:
                plot_performance_metrics(episode_rewards, episode_lengths, [rolling_avg], highest_reward)
            episode += 1
