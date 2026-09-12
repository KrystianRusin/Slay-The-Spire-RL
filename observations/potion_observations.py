import numpy as np
from util.vocabularies import potion_vocab
from observations.observation_processing import encode_potions

def get_potion_observation(game_state):
    potions = game_state.get("potions", [])
    max_potions = 5
    potion_observation = encode_potions(potions, max_potions, potion_vocab)
    
    return potion_observation