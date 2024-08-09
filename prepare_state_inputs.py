import numpy as np
from tokenizers import card_tokenizer, intent_tokenizer, monster_id_tokenizer

def prepare_state_inputs(state):
    # Default values for missing data
    default_card = [0, 0, 0, -1, -1, -1, 0, 0, -1, 0]
    default_monster = [0, 0, 0, 0, 0, 0, -1, 0, 0] + [0] * 20  # Removed name and used monster ID

    game_state = state.get('game_state', {})
    combat_state = game_state.get('combat_state', {})

    # Extract player state
    player_state = combat_state.get('player', None)
    if player_state:
        current_hp = player_state.get('current_hp', 0)
        max_hp = player_state.get('max_hp', 0)
        block = player_state.get('block', 0)
        energy = player_state.get('energy', 0)
    else:
        current_hp = max_hp = block = energy = 0

    player_input = [current_hp, max_hp, block, energy]
    player_input = np.array(player_input).reshape(1, -1)

    # Prepare hand input
    hand_state = combat_state.get('hand', [])
    hand_input = [default_card] * 10
    for i, card in enumerate(hand_state):
        hand_input[i] = [
            card.get('exhausts', 0),
            card.get('is_playable', 0),
            card.get('cost', 0),
            card_tokenizer.texts_to_sequences([card.get('id', '')])[0][0] if card.get('id', '') else 0,
            card.get('id', 0),
            card.get('type', 0),
            card.get('ethereal', 0),
            card.get('upgrades', 0),
            card.get('rarity', 0),
            card.get('has_target', 0)
        ]
    hand_input = np.array(hand_input).reshape(1, 10, -1)

    # Prepare deck input
    deck_state = game_state.get('deck', [])
    deck_input = [default_card] * len(deck_state)
    for i, card in enumerate(deck_state):
        deck_input[i] = [
            card.get('exhausts', 0),
            card.get('is_playable', 0),
            card.get('cost', 0),
            card_tokenizer.texts_to_sequences([card.get('id', '')])[0][0] if card.get('id', '') else 0,
            card.get('id', 0),
            card.get('type', 0),
            card.get('ethereal', 0),
            card.get('upgrades', 0),
            card.get('rarity', 0),
            card.get('has_target', 0)
        ]
    deck_input = np.array(deck_input).reshape(1, len(deck_input), -1)

    # Prepare monster input
    monster_state = combat_state.get('monsters', [])
    monster_input = [default_monster] * 5
    for i, monster in enumerate(monster_state):
        monster_powers = monster.get('powers', [0] * 20)
        if len(monster_powers) < 20:
            monster_powers = monster_powers + [0] * (20 - len(monster_powers))
        elif len(monster_powers) > 20:
            monster_powers = monster_powers[:20]

        monster_input[i] = [
            int(monster.get('is_gone', 0)), monster.get('move_hits', 0), monster.get('move_base_damage', 0),
            int(monster.get('half_dead', 0)), monster.get('move_adjusted_damage', 0), monster.get('max_hp', 0),
            intent_tokenizer.texts_to_sequences([monster.get('intent', 'UNKNOWN')])[0][0] if monster.get('intent', 'UNKNOWN') else 0,
            monster_id_tokenizer.texts_to_sequences([monster.get('id', '')])[0][0] if monster.get('id', '') else 0,
            monster.get('current_hp', 0), monster.get('block', 0)
        ] + monster_powers
    
    monster_input = np.array(monster_input).reshape(1, 5, -1)

    return [player_input, hand_input, deck_input, monster_input]
