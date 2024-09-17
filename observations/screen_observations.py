import numpy as np
from util.tokenizers import screen_type_tokenizer, map_symbol_tokenizer, relic_tokenizer, potion_tokenizer, card_tokenizer, rest_tokenizer, reward_type_tokenizer, event_id_tokenizer, card_tokenizer, card_type_tokenizer, card_rarity_tokenizer
from observation_processing import tokenize_card

import numpy as np

MAX_SCREEN_OBSERVATION_SIZE = 50  # Example size, set this to the largest observation size needed

def get_screen_observation(game_state):
    screen_type = game_state.get("screen_type", "NONE")
    screen_state = game_state.get("screen_state", {})
    
    # Tokenize the screen_type
    screen_type_token = screen_type_tokenizer.texts_to_sequences([screen_type])[0][0] if screen_type_tokenizer.texts_to_sequences([screen_type]) else 0

    # Get the observation from the appropriate screen handler
    if screen_type == "SHOP_SCREEN":
        observation = handle_shop_screen(screen_state, screen_type_token)
    elif screen_type == "REST":
        observation = handle_rest_screen(screen_state, screen_type_token)
    elif screen_type == "MAP":
        observation = handle_map_screen(screen_state, screen_type_token)
    elif screen_type == "HAND_SELECT":
        observation = handle_hand_select_screen(screen_state, screen_type_token)
    elif screen_type == "EVENT":
        observation = handle_event_screen(screen_state, screen_type_token)
    elif screen_type == "CHEST":
        observation = handle_chest_screen(screen_state, screen_type_token)
    elif screen_type == "COMBAT_REWARD":
        observation = handle_combat_reward_screen(screen_state, screen_type_token)
    elif screen_type == "CARD_REWARD":
        observation = handle_card_reward_screen(screen_state, screen_type_token)
    elif screen_type == "BOSS_REWARD":
        observation = handle_boss_reward_screen(screen_state, screen_type_token)
    elif screen_type == "GRID":
        observation = handle_grid_screen(screen_state, screen_type_token)
    else:
        observation = handle_default_screen(screen_state, screen_type_token)

    # Pad the observation to the maximum size
    if len(observation) < MAX_SCREEN_OBSERVATION_SIZE:
        padding = np.zeros(MAX_SCREEN_OBSERVATION_SIZE - len(observation), dtype=np.float32)
        observation = np.concatenate([observation, padding])

    return observation



def handle_shop_screen(screen_state, screen_type_token):
    max_cards = 6
    max_potions = 3
    max_relics = 3

    # Cards
    cards_observation = []
    for card in screen_state.get("cards", [])[:max_cards]:
        card_observation = [
            float(card["cost"]),
            float(card["price"]),
            float(card_tokenizer.texts_to_sequences([card["id"]])[0][0] if card_tokenizer.texts_to_sequences([card["id"]]) else 0)
        ]
        cards_observation.append(card_observation)
    while len(cards_observation) < max_cards:
        cards_observation.append([0.0, 0.0, 0.0])

    # Potions
    potions_observation = []
    for potion in screen_state.get("potions", [])[:max_potions]:
        potion_observation = [
            float(potion["price"]),
            float(potion_tokenizer.texts_to_sequences([potion["id"]])[0][0] if potion_tokenizer.texts_to_sequences([potion["id"]]) else 0)
        ]
        potions_observation.append(potion_observation)
    while len(potions_observation) < max_potions:
        potions_observation.append([0.0, 0.0])

    # Relics
    relics_observation = []
    for relic in screen_state.get("relics", [])[:max_relics]:
        relic_observation = [
            float(relic["price"]),
            float(relic_tokenizer.texts_to_sequences([relic["id"]])[0][0] if relic_tokenizer.texts_to_sequences([relic["id"]]) else 0),
            float(relic.get("counter", -1))
        ]
        relics_observation.append(relic_observation)
    while len(relics_observation) < max_relics:
        relics_observation.append([0.0, 0.0, -1.0])

    # Other shop-specific observations
    purge_cost = float(screen_state.get("purge_cost", 0.0))
    purge_available = float(screen_state.get("purge_available", 0.0))

    return np.array([
        screen_type_token,
        purge_cost,
        purge_available,
        *np.array(cards_observation).flatten(),
        *np.array(potions_observation).flatten(),
        *np.array(relics_observation).flatten()
    ], dtype=np.float32)


def handle_rest_screen(screen_state, screen_type_token):
    has_rested = float(screen_state.get("has_rested", 0.0))
    rest_options = screen_state.get("rest_options", [])
    
    # Convert rest options into a fixed-size observation space
    max_rest_options = 3
    rest_options_observation = [
        rest_tokenizer.texts_to_sequences([option])[0][0] if rest_tokenizer.texts_to_sequences([option]) else 0
        for option in rest_options[:max_rest_options]
    ]
    while len(rest_options_observation) < max_rest_options:
        rest_options_observation.append(0.0)

    return np.array([screen_type_token, has_rested, *rest_options_observation], dtype=np.float32)

def handle_map_screen(screen_state, screen_type_token):
    first_node_chosen = float(screen_state.get("first_node_chosen", 0.0))
    current_node = screen_state.get("current_node", {"symbol": "?", "x": 0, "y": 0})
    next_nodes = screen_state.get("next_nodes", [])

    # Encode current node
    current_node_observation = [
        map_symbol_tokenizer.texts_to_sequences([current_node["symbol"]])[0][0] if map_symbol_tokenizer.texts_to_sequences([current_node["symbol"]]) else 0,
        float(current_node["x"]),
        float(current_node["y"])
    ]

    # Encode next nodes (maximum of 3 for example)
    max_next_nodes = 3
    next_nodes_observation = []
    for node in next_nodes[:max_next_nodes]:
        node_observation = [
            map_symbol_tokenizer.texts_to_sequences([node["symbol"]])[0][0] if map_symbol_tokenizer.texts_to_sequences([node["symbol"]]) else 0,
            float(node["x"]),
            float(node["y"])
        ]
        next_nodes_observation.append(node_observation)
    while len(next_nodes_observation) < max_next_nodes:
        next_nodes_observation.append([0.0, 0.0, 0.0])

    boss_available = float(screen_state.get("boss_available", 0.0))

    return np.array([screen_type_token, first_node_chosen, *current_node_observation, boss_available, *np.array(next_nodes_observation).flatten()], dtype=np.float32)

def handle_hand_select_screen(screen_state):
    # Extract the max number of cards that can be selected
    max_cards = screen_state.get("max_cards", 0)
    can_pick_zero = float(screen_state.get("can_pick_zero", False))
    
    # Selected cards observation
    selected_cards = screen_state.get("selected", [])
    max_selected_cards = 10  # Assuming a max of 10 selected cards for flexibility
    selected_observation = []
    
    for card in selected_cards[:max_selected_cards]:
        card_observation = tokenize_card(card)
        selected_observation.append(card_observation)
    
    while len(selected_observation) < max_selected_cards:
        selected_observation.append([0.0] * 8)
    
    selected_observation = np.array(selected_observation, dtype=np.float32)

    # Hand observation
    hand_cards = screen_state.get("hand", [])
    max_hand_size = 10
    hand_observation = []
    
    for card in hand_cards[:max_hand_size]:
        card_observation = tokenize_card(card)
        hand_observation.append(card_observation)
    
    while len(hand_observation) < max_hand_size:
        hand_observation.append([0.0] * 8)
    
    hand_observation = np.array(hand_observation, dtype=np.float32)

    # Combine the observations for the selected cards and the hand
    # Flattened to keep consistency with the observation space
    combined_observation = np.concatenate([
        np.array([float(max_cards), can_pick_zero], dtype=np.float32),  # Meta information
        selected_observation.flatten(),  # Flatten the selected cards array
        hand_observation.flatten()  # Flatten the hand array
    ])
    
    return combined_observation

def handle_event_screen(screen_state):
    # Tokenize the event_id
    event_id_token = event_id_tokenizer.texts_to_sequences([screen_state.get("event_id", "UNKNOWN")])[0][0] if event_id_tokenizer.texts_to_sequences([screen_state.get("event_id", "UNKNOWN")]) else 0
    
    # Process options
    options = screen_state.get("options", [])
    max_options = 5  # Assume a maximum of 5 options for flexibility
    options_observation = []
    
    for option in options[:max_options]:
        choice_index = option.get("choice_index", 0)
        disabled = float(option.get("disabled", False))
        options_observation.append([float(choice_index), disabled])
    
    # Pad the options observation if fewer than max_options
    while len(options_observation) < max_options:
        options_observation.append([0.0, 0.0])
    
    options_observation = np.array(options_observation, dtype=np.float32).flatten()
    
    # Combine event_id with options observation
    event_observation = np.concatenate([
        np.array([float(event_id_token)], dtype=np.float32),  # Event ID as a float
        options_observation  # Flattened options array
    ])
    
    return event_observation

def handle_combat_reward_screen(screen_state, screen_type_token):
    # Retrieve rewards from the screen state
    rewards = screen_state.get("rewards", [])
    max_rewards = 5  # Define a maximum of 5 rewards
    
    reward_observation = []
    
    # Process each reward
    for reward in rewards[:max_rewards]:
        reward_type = reward.get("reward_type", "UNKNOWN")
        reward_token = reward_type_tokenizer.texts_to_sequences([reward_type])[0][0] if reward_type_tokenizer.texts_to_sequences([reward_type]) else 0
        reward_observation.append(float(reward_token))
    
    # Pad with zeros if fewer than max_rewards
    while len(reward_observation) < max_rewards:
        reward_observation.append(0.0)
    
    # Convert to a numpy array for the observation and include the screen_type_token
    return np.array([screen_type_token, *reward_observation], dtype=np.float32)

def handle_chest_screen(screen_state, screen_type_token):
    # Retrieve the chest_open flag from the screen state
    chest_open = float(screen_state.get("chest_open", 0.0))
    
    # Return an array with the screen_type_token and the chest_open flag
    return np.array([screen_type_token, chest_open], dtype=np.float32)

def handle_card_reward_screen(screen_state, screen_type_token):
    max_cards = 3
    
    # Get the list of cards from the screen state
    cards = screen_state.get("cards", [])
    
    # Initialize the card observation array
    card_observation = []
    
    for card in cards[:max_cards]:  # Truncate if more than max_cards
        card_name_token = card_tokenizer.texts_to_sequences([card["name"]])[0][0] if card_tokenizer.texts_to_sequences([card["name"]]) else 0
        card_type_token = card_type_tokenizer.texts_to_sequences([card["type"]])[0][0] if card_type_tokenizer.texts_to_sequences([card["type"]]) else 0
        card_rarity_token = card_rarity_tokenizer.texts_to_sequences([card["rarity"]])[0][0] if card_rarity_tokenizer.texts_to_sequences([card["rarity"]]) else 0
        
        # Handle card cost
        cost = card.get("cost", None)
        if cost is None:
            card_cost = -1
        elif cost == 'X':
            card_cost = -2
        else:
            card_cost = float(cost)
        
        # Construct card observation
        card_data = [
            float(card["exhausts"]),
            float(card["is_playable"]),
            card_cost,
            float(card_name_token),
            float(card_type_token),
            float(card_rarity_token),
            float(card["ethereal"]),
            float(card["upgrades"] > 0)
        ]
        
        card_observation.append(card_data)
    
    # Pad the card observation if fewer than max_cards
    while len(card_observation) < max_cards:
        card_observation.append([0.0] * 8)
    
    # Convert card observation to a numpy array
    card_observation = np.array(card_observation, dtype=np.float32)
    
    # Retrieve bowl_available and skip_available flags
    bowl_available = float(screen_state.get("bowl_available", False))
    skip_available = float(screen_state.get("skip_available", False))
    
    # Return the flattened observation
    return np.concatenate((
        [screen_type_token, bowl_available, skip_available],  # Screen type token and flags
        card_observation.flatten()  # Flatten the card observation to a 1D array
    ))

def handle_boss_reward_screen(screen_state, screen_type_token):
    max_relics = 3
    
    # Get the list of relics from the screen state
    relics = screen_state.get("relics", [])
    
    # Initialize the relic observation array
    relic_observation = []
    
    for relic in relics[:max_relics]:  # Truncate if more than max_relics
        relic_name_token = relic_tokenizer.texts_to_sequences([relic["name"]])[0][0] if relic_tokenizer.texts_to_sequences([relic["name"]]) else 0
        relic_counter = float(relic.get("counter", -1))
        
        # Construct relic observation
        relic_data = [float(relic_name_token), relic_counter]
        
        relic_observation.append(relic_data)
    
    # Pad the relic observation if fewer than max_relics
    while len(relic_observation) < max_relics:
        relic_observation.append([0.0, 0.0])
    
    # Convert relic observation to a numpy array and flatten it
    relic_observation = np.array(relic_observation, dtype=np.float32).flatten()
    
    # Return the flattened observation
    return np.concatenate(([screen_type_token], relic_observation))

def handle_default_screen(screen_state, screen_type_token):
    # Default to just returning the screen_type_token and a zeroed observation
    # The size of the observation can be a simple placeholder, here we'll assume a length of 5
    # This can be adjusted depending on how you want to handle other default cases in the future
    default_observation = np.zeros(5, dtype=np.float32)  # Adjust size as needed

    # Return the screen_type_token followed by default values
    return np.concatenate(([screen_type_token], default_observation))

def handle_grid_screen(screen_state, screen_type_token):
    # Define maximum number of cards and selected cards for this screen
    max_cards = 39
    max_selected_cards = 5
    
    # Tokenize and process the `cards` list
    cards_observation = []
    for card in screen_state.get("cards", [])[:max_cards]:  # Process up to max_cards
        card_observation = tokenize_card(card)  # Assuming `tokenize_card` is used here
        cards_observation.append(card_observation)
    
    # Pad if fewer than max_cards
    while len(cards_observation) < max_cards:
        cards_observation.append([0.0] * 8)  # Assuming each card observation has 8 features
    
    # Flatten the cards observation
    cards_observation = np.array(cards_observation, dtype=np.float32).flatten()
    
    # Process the `selected_cards` list, max 5 selected cards
    selected_cards_observation = []
    for card in screen_state.get("selected_cards", [])[:max_selected_cards]:
        card_observation = tokenize_card(card)  # Assuming `tokenize_card` is used here
        selected_cards_observation.append(card_observation)
    
    # Pad if fewer than max_selected_cards
    while len(selected_cards_observation) < max_selected_cards:
        selected_cards_observation.append([0.0] * 8)  # Assuming each card observation has 8 features
    
    # Flatten the selected cards observation
    selected_cards_observation = np.array(selected_cards_observation, dtype=np.float32).flatten()
    
    # Handle boolean values
    for_transform = float(screen_state.get("for_transform", False))
    confirm_up = float(screen_state.get("confirm_up", False))
    any_number = float(screen_state.get("any_number", False))
    for_upgrade = float(screen_state.get("for_upgrade", False))
    num_cards = float(screen_state.get("num_cards", 0))
    for_purge = float(screen_state.get("for_purge", False))
    
    # Combine the observations
    combined_observation = np.concatenate([
        [screen_type_token],  # Screen type token
        [for_transform, confirm_up, any_number, for_upgrade, num_cards, for_purge],  # Boolean values
        cards_observation,  # Flattened cards observation
        selected_cards_observation  # Flattened selected cards observation
    ])
    
    # Ensure the observation is padded or truncated to 50 values
    max_observation_size = 50
    if len(combined_observation) < max_observation_size:
        padding = np.zeros(max_observation_size - len(combined_observation), dtype=np.float32)
        combined_observation = np.concatenate([combined_observation, padding])
    elif len(combined_observation) > max_observation_size:
        combined_observation = combined_observation[:max_observation_size]
    
    return combined_observation