# Slay the Spire PPO Agent

This project implements a reinforcement learning agent using Maskable Proximal Policy Optimization (PPO) to play the game *Slay the Spire*. The agent interacts with the game via a socket connection through a middleman process, processes the game state, and learns to improve its actions through training.

## Table of Contents

- [Project Overview](#project-overview)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Running the Tests](#running-the-tests)
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

- **Python 3.13** (the version this project is developed and tested against)
- A running PostgreSQL instance
- *Slay the Spire* (with the [Communication Mod](https://github.com/ForgottenArbiter/CommunicationMod) installed)
- Optional: an NVIDIA GPU with a CUDA-capable driver, for GPU training

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/KrystianRusin/slAI_the_spire.git
    cd slAI_the_spire
    ```

2. Set up a virtual environment:

    ```bash
    py -3.13 -m venv venv          # On Linux/macOS: python3.13 -m venv venv
    venv\Scripts\activate          # On Linux/macOS: source venv/bin/activate
    ```

3. Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. For GPU training, reinstall torch from the PyTorch CUDA index. This step must
   come **after** step 3: resolving the other packages pulls the CPU wheel from
   PyPI over the top of a CUDA build, so installing torch first does not stick.

    ```bash
    pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128
    ```

   `cu128` suits an Ampere card (RTX 30xx) on a recent driver; pick the index
   matching your CUDA runtime from [pytorch.org](https://pytorch.org/get-started/locally/).
   Skipping this step leaves a working CPU build. Verify with:

    ```bash
    python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
    ```

5. Install the [Communication Mod](https://github.com/ForgottenArbiter/CommunicationMod) for *Slay the Spire*

6. Copy `.env.example` to `.env` and fill in your database URL:

    ```bash
    cp .env.example .env           # On Windows: copy .env.example .env
    ```

   `DATABASE_URL` is the only environment variable the code reads:

    ```
    DATABASE_URL=postgresql://<username>:<password>@localhost:<port>/<db_name>
    ```

   Tables are created automatically on the first run of `main.py`; you only need
   the database itself to exist and the credentials to be valid.

## Usage

### Running the Agent

Ensure that *Slay the Spire* is running and that the Communication Mod is set to run middleman_process.py

Start the middleman_process.py from the Communication Mod

Run the main script to start training the PPO agent:

This will start the training loop, during which the agent will interact with the game, receive game states, make decisions, and learn over time.

The model will periodically save its progress to a file named maskable_ppo_slay_the_spire.zip.

In order to use multiple Slay The Spire Environments, first ensure that all games are running `middleman_process.py` through the communication mod

Then edit `num_envs` in the main process and set it equal to the number of game instances you have open, then just run `main.py`

## Running the Tests

The tests need no database, no GPU and no running copy of the game: they work
from committed game-state fixtures. From a clean checkout, with the virtual
environment active:

```bash
pip install -r requirements-dev.txt
pytest
```

`requirements-dev.txt` pulls in `requirements.txt`, so that is the only install
step. Two tests are expected to fail and are marked accordingly:

- `test_observations_conform_to_the_declared_space` checks that every
  observation component lands inside the space the environment declares for
  it. It does not today - raw HP, tokenizer IDs, map rows and negative sentinel
  costs all escape their declared bounds - and it is the specification for
  ticket 05.
- `test_an_unlisted_card_name_does_not_crash_the_pipeline` feeds in a name the
  hard-coded vocabulary does not have. That raises `IndexError` today, and is
  the specification for ticket 03.

To read either failure in full, including the report naming every component
that is out of bounds:

```bash
pytest tests/test_observation_conformance.py --runxfail
```

Both markers are strict, so once those tickets land the tests fail by
*passing*, which is the signal to delete the marker. Note what conformance
cannot see: ticket 04's dead wiring leaves the deck component all zeros and the
screen component on its default handler, both of which are comfortably in
bounds. Those are ticket 04's own criteria to test.

### Fixtures

`tests/fixtures/game_states/` holds Communication Mod payloads covering combat,
the card reward, map, shop and rest screens, game over, and the pre-run main
menu. See the README in that directory for what each one covers and how to
capture more by setting `STS_CAPTURE_DIR` before starting the middleman.

## Customization

### Training Hyperparameters

You can adjust PPO training hyperparameters such as `learning_rate`, `gamma`, and `ent_coef` inside the `main.py` and `run_env.py` files when initializing the PPO model. Note that there are two instances of the model, one is a central model defined in `main.py` which gets updated with experiences that are collected from multiple environments. The model defined in `run_env.py` is a separate environments isolated instance of the model to interact with the game in order to avoid conflicts and interference from other environments.

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

- Modifying the game to remove graphics rendering in order to reduce load on processor, which allows for an increased number of concurrent environments running on one machine
- Introducing a imitation learning approach where the agent can learn off of gamplay from another source for pre-training 
- Implementing more detailed data analysis in order to visualize things like what specific cards the agent tends to prefer, how many elites does it defeat per run, enemies it loses to the most etc.

## Acknowledgements

This project was built with a large help from the following technologies:

- **Stable Baselines3** - A set of improved reinforcement learning algorithms in Python.
- **Slay the Spire** - The game used as the environment for this project.
- **Communication Mod** - Used for enabling communication between *Slay the Spire* and the agent via socket.
  
## Notes

This project is my first attempt at creating a full RL agent on a custom environment so feel free to raise any issues on the repository if you encounter any problems
