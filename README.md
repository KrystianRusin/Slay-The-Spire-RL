# Slay the Spire PPO Agent

This project implements a reinforcement learning agent using Maskable Proximal Policy Optimization (PPO) to play the game *Slay the Spire*. The agent interacts with the game via a socket connection through a middleman process, processes the game state, and learns to improve its actions through training.

## Table of Contents

- [Project Overview](#project-overview)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Training the Agent](#training-the-agent)
- [Observation and Action Spaces](#observation-and-action-spaces)
- [Reward Function](#reward-function)
- [Model Checkpointing](#model-checkpointing)
- [Performance Metrics](#performance-metrics)
- [Next Steps](#next-steps)
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
    git clone https://github.com/KrystianRusin/Slay-The-Spire-RL.git
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

4. Install the [Communication Mod](https://github.com/ForgottenArbiter/CommunicationMod) for *Slay the Spire*
   
5. Create .env file in root directory and set database connection url to PostgreSQL database
   ```
    DATABASE_URL=postgresql://<username>:<password>@localhost:<port>/<db_name>
   ```

## Usage

### Running the Agent

Ensure that *Slay the Spire* is running and that the Communication Mod is set to run middleman_process.py

Start the middleman_process.py from the Communication Mod

Run the main script to start training the PPO agent:

This will start the training loop, during which the agent will interact with the game, receive game states, make decisions, and learn over time.

The model will periodically save its progress to a file named maskable_ppo_slay_the_spire.zip.

In order to use multiple Slay The Spire Environments, first ensure that all games are running `middleman_process.py` through the communication mod

Then edit `num_envs` in the main process and set it equal to the number of game instances you have open, then just run `main.py`

## Customization

### Training Hyperparameters

You can adjust PPO training hyperparameters such as `learning_rate`, `gamma`, and `ent_coef` inside the `main.py` and `run_env.py` files when initializing the PPO model. Note that there are two instances of the model, one is a central model defined in `main.py` which gets updated with expereinces that are collected from multiple environments. The model defined in `run_env.py` is a separate environments isolated instance of the model to interact with the game in order to avoid conflicts and interference from other environments.

### Model Checkpoints

The model will automatically save after a defined number of steps `n_steps`, increasing `n_steps` increases data available per update leading to more stability during training, but also will require more memory.

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

- The agent's performance is tracked by logging the total rewards and episode lengths. These metrics are plotted periodically during training and saved as `performance_metrics.png`.
- Other data is stored in the database and stores each game instance with a start and end time, class chosen, floor reached and how many bosses the agent was able to defeat.
- Cards that the agent chooses during card selection is also stored along with the other choices the agent had in order to develop a card ranking based on the agents preferences
- Card performance statistics are also recorded which include how many times the card is picked, the average floor the agent reaches with that card in its deck, the cards winrate and how many games the card was featured in

## Next Steps

- Setting up Docker containers or VMs through AWS EC2 to handle each game instance and the main process in order to distribute computational load and speed up training.
- Implementing more detailed data analysis in order to visualize things like what specific cards the agent tends to prefer, how many elites does it defeat per run, enemies it loses to the most etc.

## Acknowledgements

This project was built with a large help from the following technologies:

- **Stable Baselines3** - A set of improved reinforcement learning algorithms in Python.
- **Slay the Spire** - The game used as the environment for this project.
- **Communication Mod** - Used for enabling communication between *Slay the Spire* and the agent via socket.
  
## Notes

This project is my first attempt at creating a full RL agent on a custom environment so feel free to raise any issues on the repository if you encounter any problems
