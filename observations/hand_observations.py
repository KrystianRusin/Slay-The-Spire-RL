import numpy as np
from observations.observation_processing import encode_cost
from util.vocabularies import card_rarity_vocab, card_type_vocab, card_vocab

def get_hand_observation(combat_state):
    # Default to zeros if combat_state or hand is not available
    max_hand_size = 10
    hand_observation = np.zeros((max_hand_size, 8), dtype=np.float32)

    if combat_state is None:
        return hand_observation

    # Get the hand state from combat_state
    hand_state = combat_state.get("hand", [])
    hand_observation_list = []

    for card in hand_state[:max_hand_size]:  # Truncate if more than 10 cards
        card_name_token = card_vocab.id_of(card["name"])
        card_type_token = card_type_vocab.id_of(card["type"])
        card_rarity_token = card_rarity_vocab.id_of(card["rarity"])

        card_cost = encode_cost(card)

        # Construct card observation
        card_observation = [
            float(card["exhausts"]),
            float(card["is_playable"]),
            card_cost,
            card_name_token,
            card_type_token,
            card_rarity_token,
            float(card["ethereal"]),
            float(card["upgrades"] > 0)
        ]

        hand_observation_list.append(card_observation)

    # Pad with zeros if fewer than max_hand_size cards
    while len(hand_observation_list) < max_hand_size:
        hand_observation_list.append([0.0] * 8)

    hand_observation = np.array(hand_observation_list, dtype=np.float32)
    return hand_observation
