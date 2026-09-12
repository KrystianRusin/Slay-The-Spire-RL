import numpy as np
from util.vocabularies import intent_vocab, monster_id_vocab, power_vocab
from observations.observation_processing import encode_monsters

def get_monster_observation(game_state):
    max_monsters = 5
    max_monster_powers = 20

    combat_state = game_state.get("combat_state", None)
    monsters = combat_state.get("monsters", []) if combat_state else []
    
    monster_observation = encode_monsters(monsters, max_monsters, monster_id_vocab,
                                          intent_vocab, power_vocab, max_monster_powers)
    return monster_observation
