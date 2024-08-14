# Slay the Spire AI with DQN and Action Masking

This project implements an AI agent to play the game *Slay the Spire* using Deep Q-Networks (DQN) enhanced with action masking to handle invalid actions. The agent interacts with the game through a custom environment built using OpenAI's Gymnasium framework. The project aims to create a reinforcement learning model capable of playing *Slay the Spire* by making intelligent decisions based on the game state. 

## Features

- **DQN with Action Masking**: The agent is based on Deep Q-Networks (DQN) and uses action masking to prevent selecting invalid actions.
- **Custom Environment**: A custom Gymnasium environment simulates the game state, allowing the agent to interact with the game.
- **Multiple Inputs**: The agent processes multiple aspects of the game state, including player stats, hand cards, deck, monster states, and screen type.
- **Tokenization**: Various elements like card names, monster intents, and powers are tokenized for numerical processing by the neural network.
- **Reinforcement Learning**: The agent learns to play the game by interacting with the environment, receiving rewards, and optimizing its policy through Q-learning.

## Features to Implement

- **Learning Optimizations**: Although the agent can learn, the reward system needs to be expanded on so that good actions are appropriately rewarded
- **Expand Observation Space**: Other input features such as the map layout need to be incorporated into the observation space which gives the model a better understanding out routing in game

## Getting Started

### Prerequisites

- **Python 3.8+**
- **pip** (Python package installer)
- **Communication Mod** (https://github.com/ForgottenArbiter/CommunicationMod)

### Installation and usage

1. Clone this repository:

   ```bash
   git clone https://github.com/KrystianRusin/slAI_the_spire.git
   cd Slay-The-Spire-RL

2. Install Necessary Dependencies

3. Execute the middleman process through the communication mod

    - Set communication mod command path to path of middleman_process.py
    - Open Mod Menu 
    - Communication Mod -> Config
    - Start External process

4. Start main.py through terminal

    ```bash
    python main.py
