import numpy as np
from tokenizers import screen_type_tokenizer

def get_extra_info_observation(game_state):
    # Retrieve screen_type, deck size, floor, gold, and ascension level from the game state
    screen_type = game_state.get("screen_type", "NONE")
    screen_type_token = screen_type_tokenizer.texts_to_sequences([screen_type])[0][0] if screen_type_tokenizer.texts_to_sequences([screen_type]) else 0
    
    extra_info = np.array([
        len(game_state.get("deck", [])),  # Deck size
        game_state.get("floor", 0),       # Floor number
        game_state.get("gold", 0),        # Gold amount
        game_state.get("ascension_level", 0),  # Ascension level
        screen_type_token                 # Screen type token
    ], dtype=np.float32)

    return extra_info