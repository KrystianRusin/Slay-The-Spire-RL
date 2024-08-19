import numpy as np
from tokenizers import card_tokenizer, card_type_tokenizer, card_rarity_tokenizer

def get_deck_observation(game_state):
    deck = game_state.get("deck", [])
    max_deck_size = 100
    deck_observation = []

    for card in deck[:max_deck_size]:
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
        card_observation = [
            float(card["exhausts"]),
            card_cost,
            card_name_token,
            card_type_token,
            card_rarity_token,
            float(card["ethereal"]),
            float(card["upgrades"] > 0),
            float(card["has_target"]),
        ]

        deck_observation.append(card_observation)

    # Pad the observation if the deck has fewer than 100 cards
    while len(deck_observation) < max_deck_size:
        deck_observation.append([0.0] * 8)

    return np.array(deck_observation, dtype=np.float32)