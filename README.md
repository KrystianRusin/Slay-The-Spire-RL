# Slay the Spire PPO Agent

This project implements a reinforcement learning agent using Proximal Policy Optimization (PPO) to play the game *Slay the Spire*. The agent interacts with the game via a socket connection through a middleman process, processes the game state, and learns to improve its actions through training.

## Table of Contents

- [Project Overview](#project-overview)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Training the Agent](#training-the-agent)
- [Observation and Action Spaces](#observation-and-action-spaces)
- [Reward Function](#reward-function)
- [Model Checkpointing](#model-checkpointing)
- [Performance Metrics](#performance-metrics)
- [Acknowledgements](#acknowledgements)

## Project Overview

The goal of this project is to train a reinforcement learning agent to play *Slay the Spire* effectively using the PPO algorithm. The agent receives observations from the game, processes them, and selects actions to maximize cumulative rewards.

The game is treated as an environment, where the agent observes the current game state and selects valid actions such as playing cards, using potions, and making choices. The agent's performance improves over time as it receives rewards based on in-game outcomes.

## Setup and Installation

### Prerequisites

- Python 3.7+
- `virtualenv` (optional but recommended)
- *Slay the Spire* (with the [Communication Mod](https://github.com/ForgottenArbiter/CommunicationMod) installed)

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-repository/slay-the-spire-ppo.git
    cd slay-the-spire-ppo
    ```

2. Set up a virtual environment:

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Install stable-baselines3:

    ```bash
    pip install stable-baselines3
    ```

5. Install the [Communication Mod](https://github.com/ForgottenArbiter/CommunicationMod) for *Slay the Spire*

## Usage

### Running the Agent

Ensure that *Slay the Spire* is running and that the Communication Mod is set to run middleman_process.py

Start the middleman_process.py from the Communication Mod

Run the main script to start training the PPO agent:

This will start the training loop, during which the agent will interact with the game, receive game states, make decisions, and learn over time.

The model will periodically save its progress to a file named ppo_slay_the_spire.zip.

## Customization

### Training Hyperparameters

You can adjust PPO training hyperparameters such as `learning_rate`, `gamma`, and `ent_coef` inside the `main.py` file when initializing the PPO model.

### Model Checkpoints

The model will automatically save after a defined number of episodes. You can adjust the frequency of checkpoints in the training loop.

## Training the Agent

The training loop consists of the agent receiving game states via socket communication, using its policy network to decide on the next action, and then stepping through the environment. Rewards are given based on the agent's performance in the game, which helps it learn the optimal strategies over time.

- **Episodes**: Each episode consists of the agent starting from the beginning of the game and playing until the episode ends (e.g., the agent wins, loses, or reaches a terminal state).
- **Rewards**: The agent is given positive rewards for actions that progress the game (e.g., defeating monsters, acquiring relics) and negative rewards for detrimental actions (e.g., taking damage).

## Observation and Action Spaces

### Observation Space

The observation space is a dictionary composed of multiple components, each corresponding to different aspects of the game state:

- `player`: Health, block, energy, and active powers of the player.
- `hand`: Information about the cards in the player's hand.
- `monsters`: Information about the current enemies, including health, block, and intents.
- `deck`: Information about the player's deck, including card details.
- `potion`: Information about available potions.
- `relics`: Information about relics in possession.
- `screen`: Information about the current game screen (e.g., combat, shop).
- `extra_info`: Additional game information (e.g., gold, floor number).

### Action Space

The action space is a discrete space representing various in-game actions such as:

- Starting the game
- Playing a card
- Using or discarding a potion
- Proceeding to the next screen
- Making choices in events

## Reward Function

The reward function is critical for guiding the agent's learning process. Rewards are given for various in-game achievements and penalized for undesirable outcomes:

- **Positive Rewards**: Defeating monsters, using potions, advancing floors, acquiring relics, and adding cards to the deck.
- **Negative Rewards**: Taking damage, discarding valuable resources, or losing gold.

The reward function is designed to incentivize the agent to learn optimal strategies over time.

## Model Checkpointing

The model is saved after a specified number of episodes during training. By default, the model is saved after each episode to a file called `ppo_slay_the_spire.zip`. You can adjust this frequency in the main script to optimize for training performance.

## Performance Metrics

The agent's performance is tracked by logging the total rewards and episode lengths. These metrics are plotted periodically during training and saved as `performance_metrics.png`.

## Acknowledgements

This project was built using the following libraries:

- **Stable Baselines3** - A set of improved reinforcement learning algorithms in Python.
- **Slay the Spire** - The game used as the environment for this project.
- **Communication Mod** - Used for enabling communication between *Slay the Spire* and the agent via socket.
  
## Notes

This project is my first attempt at creating a full RL agent on a custom environment so feel free to raise any issues on the repository if you encounter any problems
