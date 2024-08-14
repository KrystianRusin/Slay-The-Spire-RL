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

def save_metrics(episode_rewards, rolling_avg_rewards, episode_lengths):
    plt.figure(figsize=(12, 6))

    plt.subplot(3, 1, 1)
    plt.plot(range(0, len(episode_rewards), 2), episode_rewards[::2], label="Cumulative Reward per Episode")
    plt.title("Cumulative Rewards per Episode")
    plt.xlabel("Episode (Even Numbers)")
    plt.ylabel("Cumulative Reward")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(range(0, len(rolling_avg_rewards), 2), rolling_avg_rewards[::2], label="Rolling Average Reward", color='orange')
    plt.title("Rolling Average Rewards")
    plt.xlabel("Episode (Even Numbers)")
    plt.ylabel("Rolling Average Reward")
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(range(0, len(episode_lengths), 2), episode_lengths[::2], label="Episode Lengths", color='green')
    plt.title("Episode Lengths")
    plt.xlabel("Episode (Even Numbers)")
    plt.ylabel("Length (timesteps)")
    plt.legend()

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
    policy = EpsGreedyQPolicy(eps=0.75)
    agent = MaskedDQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=1000, target_model_update=1e-2, policy=policy)
    agent.compile(Adam(learning_rate=1e-3), metrics=['mae'])

    if os.path.exists(WEIGHTS_FILE):
        print(f"Loading weights from {WEIGHTS_FILE}")
        agent.load_weights(WEIGHTS_FILE)

    episode_rewards = []
    rolling_avg_rewards = []
    episode_lengths = []

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

            state_inputs = prepare_state_inputs(env.state)
            time.sleep(0.5)

            action = agent.forward({'state': state_inputs, 'invalid_action_mask': env.get_invalid_action_mask()})
            chosen_command = env.actions[action]

            print("Chosen action index:", action)
            print("Chosen action:", chosen_command)

            state, reward, done, info = env.step(action, game_state)
            client_socket.sendall(chosen_command.encode('utf-8'))

            agent.backward(reward, terminal=done)

            total_reward += reward
            episode_length += 1

        # Store the cumulative reward and episode length after each episode
        episode_rewards.append(total_reward)
        rolling_avg_rewards.append(np.mean(episode_rewards[-100:]))
        episode_lengths.append(episode_length)

        # Save the performance metrics for even-numbered episodes only
        if len(episode_rewards) % 2 == 0:
            save_metrics(episode_rewards, rolling_avg_rewards, episode_lengths)

        print("Starting a new episode...")
        agent.save_weights(WEIGHTS_FILE, overwrite=True)

    client_socket.close()

if __name__ == "__main__":
    main()
