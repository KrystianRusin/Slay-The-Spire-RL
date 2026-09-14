import torch as th
from collections import deque
from sb3_contrib.ppo_mask import MaskablePPO
from slay_the_spire_env import SlayTheSpireEnv
from model.custom_rollout_buffer import CustomRolloutBuffer
from model.policy_follower import PolicyFollower
from model.rollout_codec import encode_rollout
from util.communication import GameConnection, handle_end_of_episode
from util.plotting import plot_performance_metrics
from util.data_processor import process_game_state
import json
from util.game_over_tracking import update_game_stats_on_game_over
from util.card_tracking import track_card_performance

def hand_off_rollout(rollout_buffer, publisher):
    """Publish a completed rollout for the learner, encoded, and clear the buffer for the next one."""
    publisher.publish(encode_rollout(rollout_buffer))
    rollout_buffer.reset()

def run_environment(env_id, port, publisher, policy_source, n_steps=2048):
    """
    Function to run a single agent in a separate environment.

    The policy follows the newest weights read from policy_source, switching only between rollouts.
    """
    episode_rewards = []
    episode_lengths = []
    reward_queue = deque(maxlen=10)
    highest_reward = float('-inf')
    
    connection = GameConnection(("localhost", port))

    # Initialize the environment
    env = SlayTheSpireEnv({})
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    model = MaskablePPO("MultiInputPolicy", env, ent_coef=0.03, gamma=0.97, learning_rate=0.0003, clip_range=0.3, verbose=1, device=device)
    follower = PolicyFollower(model.policy, policy_source)
    follower.wait_for_first()
    print(f"Environment {env_id}: Running policy version {follower.version}")

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
            print(f"Environment {env_id}: Game State Received")

            env.update_game_state(game_state)
            obs = env.flatten_observation(game_state)
            obs_tensor = {key: th.tensor(value, dtype=th.float32).unsqueeze(0).to(device) for key, value in obs.items()}

            action_mask = env.get_valid_action_mask(game_state)
            action_mask_tensor = th.tensor(action_mask, dtype=th.bool).unsqueeze(0).to(device)
            obs_numpy = {key: value.cpu().numpy() for key, value in obs_tensor.items()}
            action_mask_numpy = action_mask_tensor.cpu().numpy()

            action, _states = model.predict(obs_numpy, action_masks=action_mask_numpy)
            action = int(action)
            chosen_command = env.actions[action].text
            try:
                connection.send(chosen_command)
            except ConnectionError as e:
                print(f"Environment {env_id}: {e}; not recording the action")
                continue

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
                hand_off_rollout(rollout_buffer, publisher)

                collected_under = follower.version
                switched = follower.update()
                print(f"Environment {env_id}: Published a rollout collected under policy version {collected_under}; "
                      f"the learner is at version {follower.learner_version}")
                if switched:
                    print(f"Environment {env_id}: Running policy version {follower.version}")
            if done:
                screen_state = game_state['game_state'].get('screen_state', {})
                victory = screen_state.get('victory', False)
                floor_reached = game_state['game_state'].get('floor', 0)

                # Update the game stats and card performance before sending any commands
                update_game_stats_on_game_over(game_state, game_id, total_reward)
                track_card_performance(game_state['game_state'], floor_reached, victory)

        episode_rewards.append(total_reward)
        episode_lengths.append(episode_length)
        reward_queue.append(total_reward)
        highest_reward = max(highest_reward, total_reward)
        rolling_avg = sum(reward_queue) / len(reward_queue) if reward_queue else 0

        if episode % 10 == 0:
            plot_performance_metrics(episode_rewards, episode_lengths, [rolling_avg], highest_reward)

        episode += 1
  
        handle_end_of_episode(connection)

    connection.close()
