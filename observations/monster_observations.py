import numpy as np
from tokenizers import monster_id_tokenizer, intent_tokenizer, power_tokenizer
from observation_processing import tokenize_monsters

def get_monster_observation(game_state):
    max_monsters = 5
    max_monster_powers = 20

    combat_state = game_state.get("combat_state", None)
    monsters = combat_state.get("monsters", []) if combat_state else []
    
    monster_observation = tokenize_monsters(monsters, max_monsters, monster_id_tokenizer, intent_tokenizer, power_tokenizer, max_monster_powers)
    return monster_observation
