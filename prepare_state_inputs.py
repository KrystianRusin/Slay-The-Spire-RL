import numpy as np
from tokenizers import card_tokenizer, intent_tokenizer, monster_id_tokenizer, card_type_tokenizer, card_rarity_tokenizer, screen_type_tokenizer, map_symbol_tokenizer, relic_tokenizer

def prepare_state_inputs(state):
    # Default values for missing data
    default_card = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    default_monster = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    default_map_room = [0, 0, 0, 0, 0, 0]  # For symbol, x, y, child_x, child_y, is_current_room
    default_relic = [0, -1]

    # Check if 'game_state' exists
    game_state = state.get('game_state', None)
    if not game_state:
        # If game_state doesn't exist, return default inputs
        return [np.zeros((1, 4)), np.zeros((1, 10, 9)), np.zeros((1, 1, 9)), np.zeros((1, 5, 10)), np.zeros((1, 51, 6)), np.zeros((1, 1)), np.zeros((1, 1, 2))]

    # Extract player state
    player_state = game_state.get('player', None)
    current_hp = player_state.get('current_hp', 0) if player_state else 0
    max_hp = player_state.get('max_hp', 0) if player_state else 0
    block = player_state.get('block', 0) if player_state else 0
    energy = player_state.get('energy', 0) if player_state else 0

    player_input = np.array([current_hp, max_hp, block, energy]).reshape(1, -1)

    # Prepare hand input
    hand_state = game_state.get('hand', [])
    hand_input = [default_card] * 10
    for i, card in enumerate(hand_state):
        hand_input[i] = [
            card.get('exhausts', 0),
            card.get('is_playable', 0),
            card.get('cost', 0),
            card_tokenizer.texts_to_sequences([card.get('name', '')])[0][0] if card.get('name', '') else 0,
            card_type_tokenizer.texts_to_sequences([card.get('type', '')])[0][0] if card.get('type', '') else 0,
            card.get('ethereal', 0),
            card.get('upgrades', 0),
            card_rarity_tokenizer.texts_to_sequences([card.get('rarity', '')])[0][0] if card.get('rarity', '') else 0,
            card.get('has_target', 0)
        ]
    hand_input = np.array(hand_input).reshape(1, 10, -1)

    # Prepare deck input
    deck_state = game_state.get('deck', [])
    if len(deck_state) == 0:
        deck_input = np.zeros((1, 1, 9))  # Handle empty deck case
    else:
        deck_input = np.zeros((1, len(deck_state), 9))  # Allocate space based on actual deck size
        for i, card in enumerate(deck_state):
            deck_input[0, i] = [
                card.get('exhausts', 0),
                card.get('is_playable', 0),
                card.get('cost', 0),
                card_tokenizer.texts_to_sequences([card.get('name', '')])[0][0] if card.get('name', '') else 0,
                card_type_tokenizer.texts_to_sequences([card.get('type', '')])[0][0] if card.get('type', '') else 0,
                card.get('ethereal', 0),
                card.get('upgrades', 0),
                card_rarity_tokenizer.texts_to_sequences([card.get('rarity', '')])[0][0] if card.get('rarity', '') else 0,
                card.get('has_target', 0)
            ]

    # Prepare monster input
    monster_state = game_state.get('monsters', [])
    monster_input = [default_monster] * 5
    for i, monster in enumerate(monster_state):
        monster_input[i] = [
            int(monster.get('is_gone', 0)), monster.get('move_hits', 0), monster.get('move_base_damage', 0),
            int(monster.get('half_dead', 0)), monster.get('move_adjusted_damage', 0), monster.get('max_hp', 0),
            intent_tokenizer.texts_to_sequences([monster.get('intent', '')])[0][0] if monster.get('intent', '') else 0,
            monster_id_tokenizer.texts_to_sequences([monster.get('id', '')])[0][0] if monster.get('id', '') else 0,
            monster.get('current_hp', 0), monster.get('block', 0)
        ]
    monster_input = np.array(monster_input).reshape(1, 5, -1)

    # Prepare screen type input
    screen_type = game_state.get('screen_type', 'NONE')
    screen_type_input = screen_type_tokenizer.texts_to_sequences([screen_type])[0][0] if screen_type else 0
    screen_type_input = np.array([screen_type_input]).reshape(1, 1)

    # Prepare map input
    map_state = game_state.get('map', [])
    map_input = [default_map_room] * 51
    for i, room in enumerate(map_state[:51]):  # Only consider the first 51 rooms
        symbol = room.get('symbol', '')
        room_symbol_sequence = map_symbol_tokenizer.texts_to_sequences([symbol])

        if len(room_symbol_sequence) > 0 and len(room_symbol_sequence[0]) > 0:
            room_symbol = room_symbol_sequence[0][0]
        else:
            room_symbol = 0  # Default to 0 if the symbol is not recognized or not tokenized

        room_x = room.get('x', 0)
        room_y = room.get('y', 0)
        child_x = room.get('children', [{}])[0].get('x', 0)  # Default to 0 if no children or no x
        child_y = room.get('children', [{}])[0].get('y', 0)  # Default to 0 if no children or no y
        is_current_room = 1 if i == len(map_state) - 1 else 0  # Last room in the list might be the current room
        map_input[i] = [room_symbol, room_x, room_y, child_x, child_y, is_current_room]

    map_input = np.array(map_input).reshape(1, 51, -1)

    # Prepare relics input
    relic_state = game_state.get('relics', [])
    if len(relic_state) == 0:
        relic_input = np.zeros((1, 1, 2))  # Handle empty relic case
    else:
        relic_input = np.zeros((1, len(relic_state), 2))  # Allocate space based on actual relics size
        for i, relic in enumerate(relic_state):
            relic_input[0, i] = [
                relic_tokenizer.texts_to_sequences([relic.get('name', '')])[0][0] if relic.get('name', '') else 0,
                relic.get('counter', -1)
            ]

    return [player_input, hand_input, deck_input, monster_input, map_input, screen_type_input, relic_input]
