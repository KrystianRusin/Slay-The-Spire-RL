import matplotlib.pyplot as plt

def plot_performance_metrics(episode_rewards, episode_lengths, rolling_avg_rewards, highest_reward, save_path="performance_metrics.png"):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

    ax1.plot(range(len(episode_rewards)), episode_rewards, 'b-', label='Reward')
    ax1.axhline(y=highest_reward, color='r', linestyle='--', label=f'Highest Reward: {highest_reward}')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Episode Rewards')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(range(len(rolling_avg_rewards)), rolling_avg_rewards, 'g--', label='Rolling Avg Reward (Last 10)')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Rolling Avg Reward')
    ax2.set_title('Rolling Average of Rewards')
    ax2.legend()
    ax2.grid(True)

    ax3.plot(range(len(episode_lengths)), episode_lengths, 'r-', label='Episode Length')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Episode Length')
    ax3.set_title('Episode Lengths')
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, format='png')
    plt.close(fig)

    print(f"Performance metrics saved to {save_path}")
