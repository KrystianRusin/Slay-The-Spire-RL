import numpy as np
from observations.observation_processing import encode_cost
from util.vocabularies import card_rarity_vocab, card_type_vocab, card_vocab

def get_deck_observation(game_state):
    deck = game_state.get("deck", [])
    max_deck_size = 100
    deck_observation = []

    for card in deck[:max_deck_size]:
        card_name_token = card_vocab.id_of(card["name"])
        card_type_token = card_type_vocab.id_of(card["type"])
        card_rarity_token = card_rarity_vocab.id_of(card["rarity"])

        card_cost = encode_cost(card)

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