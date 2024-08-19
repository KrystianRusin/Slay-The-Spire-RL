import numpy as np
from tokenizers import card_tokenizer, card_rarity_tokenizer, card_type_tokenizer

def tokenize_powers(powers, max_powers, tokenizer):
    powers_observation = []
    for power in powers[:max_powers]:
        try:
            if isinstance(power, str):
                # If power is a string, tokenize it directly
                power_token = tokenizer.texts_to_sequences([power])[0][0] if tokenizer.texts_to_sequences([power]) else 0
            elif isinstance(power, dict):
                # If power is a dictionary, extract the "name" field (or appropriate field)
                power_name = power.get('name', '')  # Default to an empty string if 'name' is missing
                power_token = tokenizer.texts_to_sequences([power_name])[0][0] if tokenizer.texts_to_sequences([power_name]) else 0
            else:
                # If it's neither a string nor a dict, return 0 as a fallback
                power_token = 0
            powers_observation.append(float(power_token))
        except IndexError as e:
            print(f"Error tokenizing power: {power}")
            print(f"Exception: {e}")
            powers_observation.append(0.0)  # Add a fallback value
    
    # Pad with zeros if there are fewer than max_powers
    while len(powers_observation) < max_powers:
        powers_observation.append(0.0)
    
    return np.array(powers_observation, dtype=np.float32)

def tokenize_potions(potions, max_potions, tokenizer):
    potion_observation = []
    
    for potion in potions[:max_potions]:
        try:
            # Tokenize the potion ID
            potion_token = tokenizer.texts_to_sequences([potion['id']])[0][0] if tokenizer.texts_to_sequences([potion['id']]) else 0
        except IndexError as e:
            print(f"Error tokenizing potion: {potion}")
            print(f"Exception: {e}")
            potion_token = 0  # Add a fallback value

        # Extract the other attributes (requires_target, can_use, can_discard)
        requires_target = float(potion.get("requires_target", False))
        can_use = float(potion.get("can_use", False))
        can_discard = float(potion.get("can_discard", False))
        
        # Add the 4 attributes to the potion observation
        potion_observation.append([float(potion_token), requires_target, can_use, can_discard])
    
    # Pad with zero arrays if there are fewer than max_potions
    while len(potion_observation) < max_potions:
        potion_observation.append([0.0, 0.0, 0.0, 0.0])
    
    # Convert to numpy array
    return np.array(potion_observation, dtype=np.float32)

def tokenize_monsters(monsters, max_monsters, monster_id_tokenizer, intent_tokenizer, power_tokenizer, max_monster_powers):
    monster_observation = []

    for monster in monsters[:max_monsters]:
        try:
            # Tokenize monster ID and intent
            monster_id_token = monster_id_tokenizer.texts_to_sequences([monster["id"]])[0][0] if monster_id_tokenizer.texts_to_sequences([monster["id"]]) else 0
            monster_intent_token = intent_tokenizer.texts_to_sequences([monster["intent"]])[0][0] if intent_tokenizer.texts_to_sequences([monster["intent"]]) else 0
        except IndexError as e:
            print(f"Error tokenizing monster: {monster}")
            print(f"Exception: {e}")
            monster_id_token, monster_intent_token = 0, 0  # Add fallback values

        # Collect monster data
        monster_data = [
            float(monster.get("is_gone", 0)),
            float(monster.get("move_hits", 0)),
            float(monster.get("move_base_damage", 0)),
            float(monster.get("half_dead", 0)),
            float(monster.get("move_adjusted_damage", 0)),
            float(monster.get("max_hp", 0)),
            float(monster.get("current_hp", 0)),
            float(monster.get("block", 0)),
            float(monster_intent_token),
            float(monster_id_token)
        ]
        
        # Tokenize monster powers
        powers = monster.get("powers", [])
        powers_observation = []
        
        for power in powers[:max_monster_powers]:
            try:
                if isinstance(power, str):
                    power_token = power_tokenizer.texts_to_sequences([power])[0][0] if power_tokenizer.texts_to_sequences([power]) else 0
                elif isinstance(power, dict):
                    power_str = str(power.get('name', ''))
                    power_token = power_tokenizer.texts_to_sequences([power_str])[0][0] if power_tokenizer.texts_to_sequences([power_str]) else 0
                else:
                    power_token = 0
            except IndexError as e:
                print(f"Error tokenizing power: {power}")
                print(f"Exception: {e}")
                power_token = 0  # Add fallback value
            powers_observation.append(float(power_token))
        
        # Pad powers if fewer than max_monster_powers
        while len(powers_observation) < max_monster_powers:
            powers_observation.append(0.0)
        
        # Combine monster data and powers observation
        monster_full_observation = monster_data + powers_observation
        monster_observation.append(monster_full_observation)
    
    # Pad the observation if fewer than max_monsters
    while len(monster_observation) < max_monsters:
        monster_observation.append([0.0] * (10 + max_monster_powers))
    
    return np.array(monster_observation, dtype=np.float32)

def tokenize_card(card):
    try:
        # Tokenize card's name, type, and rarity using existing tokenizers
        card_name_token = card_tokenizer.texts_to_sequences([card["name"]])[0][0] if card_tokenizer.texts_to_sequences([card["name"]]) else 0
        card_type_token = card_type_tokenizer.texts_to_sequences([card["type"]])[0][0] if card_type_tokenizer.texts_to_sequences([card["type"]]) else 0
        card_rarity_token = card_rarity_tokenizer.texts_to_sequences([card["rarity"]])[0][0] if card_rarity_tokenizer.texts_to_sequences([card["rarity"]]) else 0
    except IndexError as e:
        print(f"Error tokenizing card: {card}")
        print(f"Exception: {e}")
        card_name_token, card_type_token, card_rarity_token = 0, 0, 0  # Add fallback values
    
    # Handle card cost, where 'X' is represented as a special case (-2)
    cost = card.get("cost", None)
    if cost is None:
        card_cost = -1  # Unknown cost
    elif cost == 'X':
        card_cost = -2  # Special cost for 'X'
    else:
        card_cost = float(cost)
    
    # Construct card observation
    card_observation = [
        float(card.get("exhausts", 0)),      # Whether the card exhausts
        float(card.get("is_playable", 0)),   # Whether the card is playable
        card_cost,                           # Card's cost (or -1 for unknown, -2 for 'X')
        float(card_name_token),              # Tokenized card name
        float(card_type_token),              # Tokenized card type
        float(card_rarity_token),            # Tokenized card rarity
        float(card.get("ethereal", 0)),      # Whether the card is ethereal
        float(card.get("upgrades", 0))       # Number of upgrades the card has
    ]
    
    return card_observation
