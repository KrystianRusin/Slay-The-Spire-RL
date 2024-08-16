import json
import numpy as np
import socket
import time
import os
import matplotlib.pyplot as plt
from slay_the_spire_env import SlayTheSpireEnv, MaskedSlayTheSpireEnv
from masked_dqn_agent import MaskedDQNAgent
from build_model import build_model
from prepare_state_inputs import prepare_state_inputs
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy
from keras.optimizers.legacy import Adam

WEIGHTS_FILE = "dqn_weights.h5f"
METRICS_PLOT_FILE = "metrics_plot.png"

# Define epsilon parameters
EPSILON_START = 1.0  # Initial exploration rate
EPSILON_MIN = 0.01   # Minimum exploration rate
EPSILON_DECAY = 0.995  # Decay rate per episode

def save_metrics(episode_rewards, rolling_avg_rewards, episode_lengths, highest_reward, highest_reward_length):
    plt.figure(figsize=(12, 6))

    # Plot cumulative rewards per episode (even-numbered episodes only)
    plt.subplot(4, 1, 1)
    plt.plot(range(0, len(episode_rewards), 2), episode_rewards[::2], label="Cumulative Reward per Episode")
    plt.title("Cumulative Rewards per Episode")
    plt.xlabel("Episode (Even Numbers)")
    plt.ylabel("Cumulative Reward")
    plt.legend()

    # Plot rolling average rewards (even-numbered episodes only, starting after episode 10)
    plt.subplot(4, 1, 2)
    plt.plot(range(10, len(rolling_avg_rewards) * 2 + 10, 2), rolling_avg_rewards, label="Rolling Average Reward (Last 10 Episodes)", color='orange')
    plt.title("Rolling Average Rewards")
    plt.xlabel("Episode (Even Numbers)")
    plt.ylabel("Rolling Average Reward")
    plt.legend()

    # Plot episode lengths (even-numbered episodes only)
    plt.subplot(4, 1, 3)
    plt.plot(range(0, len(episode_lengths), 2), episode_lengths[::2], label="Episode Lengths", color='green')
    plt.title("Episode Lengths")
    plt.xlabel("Episode (Even Numbers)")
    plt.ylabel("Length (timesteps)")
    plt.legend()

    # Display highest reward and corresponding episode length
    plt.subplot(4, 1, 4)
    plt.text(0.5, 0.5, f"Highest Reward: {highest_reward}\nEpisode Length: {highest_reward_length}",
             horizontalalignment='center', verticalalignment='center', fontsize=12, transform=plt.gca().transAxes)
    plt.title("Highest Reward and Episode Length")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(METRICS_PLOT_FILE)
    plt.close()


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
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(("localhost", 9999))

    env = SlayTheSpireEnv({})
    env = MaskedSlayTheSpireEnv(env)
    nb_actions = env.action_space.n
    model = build_model(env.observation_space, nb_actions)
    model.summary()

    memory = SequentialMemory(limit=50000, window_length=1)
    policy = EpsGreedyQPolicy(eps=EPSILON_START)
    agent = MaskedDQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=1000, target_model_update=1e-2, policy=policy)
    agent.compile(Adam(learning_rate=1e-3), metrics=['mae'])

    if os.path.exists(WEIGHTS_FILE):
        print(f"Loading weights from {WEIGHTS_FILE}")
        agent.load_weights(WEIGHTS_FILE)

    episode_rewards = []
    rolling_avg_rewards = []
    episode_lengths = []
    epsilon = EPSILON_START  # Initialize epsilon value for decay
    previous_game_state = None  # Track the previous game state

    highest_reward = float('-inf')  # Track the highest reward
    highest_reward_length = 0  # Track the episode length for the highest reward

    episode = 0
    while True:
        done = False
        total_reward = 0
        episode_length = 0

        while not done:
            try:
                game_state = receive_full_json(client_socket)
            except json.JSONDecodeError as e:
                print(f"Failed to decode JSON: {e}")
                continue

            # Check if the current game state is identical to the previous one
            if game_state == previous_game_state:
                print("Game state has not changed. Skipping this step.")
                time.sleep(0.1)  # Small delay to avoid tight loop
                continue  # Skip this iteration

            previous_game_state = game_state  # Update the previous game state

            # Prepare state inputs for the model
            state_inputs = prepare_state_inputs(env.state)
            time.sleep(0.5)

            # Use the DQN agent to select an action
            action = agent.forward({'state': state_inputs, 'invalid_action_mask': env.get_invalid_action_mask()})
            chosen_command = env.actions[action]

            print("Chosen action index:", action)
            print("Chosen action:", chosen_command)

            # Step in the environment with the chosen action
            state, reward, done, info = env.step(action, game_state)
            client_socket.sendall(chosen_command.encode('utf-8'))

            # Store the experience and train the agent
            agent.backward(reward, terminal=done)

            total_reward += reward
            episode_length += 1

        # Store the cumulative reward and episode length after each episode
        episode_rewards.append(total_reward)
        episode_lengths.append(episode_length)

        # Calculate rolling average for the last 10 episodes, starting after episode 10
        if len(episode_rewards) > 10:
            rolling_avg_rewards.append(np.mean(episode_rewards[-10:]))

        # Update the highest reward and the corresponding episode length
        if total_reward > highest_reward:
            highest_reward = total_reward
            highest_reward_length = episode_length

        # Save the performance metrics for even-numbered episodes only
        if episode % 2 == 1:
            save_metrics(episode_rewards, rolling_avg_rewards, episode_lengths, highest_reward, highest_reward_length)

        # Epsilon decay: Apply decay after each episode
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        policy.eps = epsilon  # Update the epsilon in the policy
        print(f"Epsilon updated to: {epsilon}")

        print("Starting a new episode...")
        episode += 1
        agent.save_weights(WEIGHTS_FILE, overwrite=True)

    client_socket.close()


if __name__ == "__main__":
    main()
