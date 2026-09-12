import numpy as np
from util.vocabularies import card_rarity_vocab, card_type_vocab, card_vocab

UNKNOWN_COST = -1
X_COST = -2


def encode_cost(card):
    cost = card.get("cost", None)
    if cost is None:
        return UNKNOWN_COST
    if cost == 'X':
        return X_COST
    return float(cost)


def encode_powers(powers, max_powers, vocabulary):
    powers_observation = []
    for power in powers[:max_powers]:
        # A power arrives either as a bare name or as an object carrying one.
        name = power.get('name', '') if isinstance(power, dict) else power
        powers_observation.append(float(vocabulary.id_of(name)))

    # Pad with zeros if there are fewer than max_powers
    while len(powers_observation) < max_powers:
        powers_observation.append(0.0)

    return np.array(powers_observation, dtype=np.float32)


def encode_potions(potions, max_potions, vocabulary):
    potion_observation = []

    for potion in potions[:max_potions]:
        potion_id = vocabulary.id_of(potion['id'])

        # Extract the other attributes (requires_target, can_use, can_discard)
        requires_target = float(potion.get("requires_target", False))
        can_use = float(potion.get("can_use", False))
        can_discard = float(potion.get("can_discard", False))

        # Add the 4 attributes to the potion observation
        potion_observation.append([float(potion_id), requires_target, can_use, can_discard])

    # Pad with zero arrays if there are fewer than max_potions
    while len(potion_observation) < max_potions:
        potion_observation.append([0.0, 0.0, 0.0, 0.0])

    # Convert to numpy array
    return np.array(potion_observation, dtype=np.float32)


def encode_monsters(monsters, max_monsters, monster_id_vocab, intent_vocab,
                    power_vocab, max_monster_powers):
    monster_observation = []

    for monster in monsters[:max_monsters]:
        monster_data = [
            float(monster.get("is_gone", 0)),
            float(monster.get("move_hits", 0)),
            float(monster.get("move_base_damage", 0)),
            float(monster.get("half_dead", 0)),
            float(monster.get("move_adjusted_damage", 0)),
            float(monster.get("max_hp", 0)),
            float(monster.get("current_hp", 0)),
            float(monster.get("block", 0)),
            float(intent_vocab.id_of(monster["intent"])),
            float(monster_id_vocab.id_of(monster["id"])),
        ]

        powers_observation = encode_powers(
            monster.get("powers", []), max_monster_powers, power_vocab
        )

        # Combine monster data and powers observation
        monster_observation.append(monster_data + list(powers_observation))

    # Pad the observation if fewer than max_monsters
    while len(monster_observation) < max_monsters:
        monster_observation.append([0.0] * (10 + max_monster_powers))

    return np.array(monster_observation, dtype=np.float32)


def encode_card(card):
    return [
        float(card.get("exhausts", 0)),         # Whether the card exhausts
        float(card.get("is_playable", 0)),      # Whether the card is playable
        encode_cost(card),                      # Card's cost, or a sentinel
        float(card_vocab.id_of(card["name"])),
        float(card_type_vocab.id_of(card["type"])),
        float(card_rarity_vocab.id_of(card["rarity"])),
        float(card.get("ethereal", 0)),         # Whether the card is ethereal
        float(card.get("upgrades", 0)),         # Number of upgrades the card has
    ]
