import numpy as np
from tokenizers import potion_tokenizer
from observation_processing import tokenize_potions

def get_potion_observation(game_state):
    potions = game_state.get("potions", [])
    max_potions = 5
    potion_observation = tokenize_potions(potions, max_potions, potion_tokenizer)
    
    return potion_observation