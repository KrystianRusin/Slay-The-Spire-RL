import numpy as np
from util.tokenizers import power_tokenizer
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from observations.observation_processing import tokenize_powers

def get_player_observation(game_state):
    current_hp = game_state.get("current_hp", 0)
    max_hp = game_state.get("max_hp", 1)
    block = 0
    energy = 0

    max_powers = 20
    powers_observation = np.zeros(max_powers, dtype=np.float32)

    combat_state = game_state.get("combat_state", None)
    if combat_state:
        player_state = combat_state.get("player", {})
        block = player_state.get("block", 0)
        energy = player_state.get("energy", 0)
        powers = player_state.get("powers", [])
        powers_observation = tokenize_powers(powers, max_powers, power_tokenizer)

    player_observation = np.array([current_hp, max_hp, block, energy], dtype=np.float32)
    player_full_observation = np.concatenate([player_observation, powers_observation])

    return player_full_observation
