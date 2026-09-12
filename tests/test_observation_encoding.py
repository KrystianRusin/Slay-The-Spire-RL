"""The helpers that turn one game object into numbers."""

import copy

import pytest

from observations.observation_processing import encode_card, encode_cost, encode_powers
from observations.screen_observations import handle_shop_screen
from slay_the_spire_env import SlayTheSpireEnv
from util.vocabularies import (UNKNOWN_ID, card_vocab,
                               intent_vocab, map_symbol_vocab, monster_id_vocab,
                               potion_vocab, power_vocab, relic_vocab,
                               screen_type_vocab)

from tests.conftest import fixture_names, load_payload

NAME_COLUMN = 3  # exhausts, is_playable, cost, name, type, rarity, ...


def a_card(**overrides):
    card = {
        "name": "Body Slam", "id": "Body Slam", "type": "ATTACK",
        "rarity": "COMMON", "cost": 1, "exhausts": False, "is_playable": True,
        "ethereal": False, "upgrades": 0, "has_target": True,
    }
    card.update(overrides)
    return card


def test_a_card_is_encoded_under_its_whole_name():
    encoded = encode_card(a_card())

    assert encoded[NAME_COLUMN] == card_vocab.id_of("Body Slam")
    assert encoded[NAME_COLUMN] != card_vocab.id_of("Bash")


def test_an_unlisted_card_name_encodes_as_unknown():
    encoded = encode_card(a_card(name="Sozu's Revenge"))

    assert encoded[NAME_COLUMN] == UNKNOWN_ID
    assert encoded[NAME_COLUMN + 1] != UNKNOWN_ID, "type is still recognised"


def test_shop_cards_are_looked_up_by_name_not_by_id():
    """A card id carries a colour suffix ("Strike_R") no vocabulary holds."""
    screen_state = {
        "cards": [{"name": "Strike", "id": "Strike_R", "cost": 1, "price": 45}],
        "potions": [], "relics": [], "purge_available": True, "purge_cost": 75,
    }

    observation = handle_shop_screen(screen_state, screen_type_token=1)

    first_card_name = 5  # screen type, purge cost, purge available, cost, price
    assert observation[first_card_name] == card_vocab.id_of("Strike")


def test_powers_encode_whether_they_arrive_as_names_or_objects():
    from_names = encode_powers(["Strength", "Vulnerable"], 4, power_vocab)
    from_objects = encode_powers(
        [{"name": "Strength", "amount": 3}, {"name": "Vulnerable", "amount": 2}],
        4, power_vocab,
    )

    assert list(from_names) == list(from_objects)
    assert from_names[0] == power_vocab.id_of("Strength")
    assert list(from_names[2:]) == [0.0, 0.0], "padded to the requested width"


def test_an_unknown_cost_and_an_x_cost_keep_their_sentinels():
    assert encode_cost(a_card(cost=None)) == -1
    assert encode_cost(a_card(cost="X")) == -2
    assert encode_cost(a_card(cost=2)) == 2


def test_an_unlisted_card_name_does_not_crash_the_pipeline():
    """What a modded card, or one added by a game update, does to us."""
    payload = copy.deepcopy(load_payload("combat_with_monsters"))
    payload["game_state"]["combat_state"]["hand"][0]["name"] = "Sozu's Revenge"

    observation = SlayTheSpireEnv(payload).flatten_observation(payload)

    name_column = 3  # exhausts, is_playable, cost, name, ...
    assert observation["hand"][0][name_column] == UNKNOWN_ID
    assert observation["hand"][0][name_column + 1] != UNKNOWN_ID, "the card's type survives"


def names_in(payload):
    """Every name in one payload, paired with the vocabulary that should hold it."""
    game_state = payload.get("game_state")
    if not game_state:
        return []

    found = [(screen_type_vocab, game_state["screen_type"])]
    for card in game_state["deck"]:
        found.append((card_vocab, card["name"]))
    for relic in game_state["relics"]:
        found.append((relic_vocab, relic["name"]))
    for potion in game_state["potions"]:
        found.append((potion_vocab, potion["id"]))
    for node in game_state["map"]:
        found.append((map_symbol_vocab, node["symbol"]))

    combat = game_state.get("combat_state") or {}
    for pile in ("hand", "draw_pile", "discard_pile", "exhaust_pile"):
        for card in combat.get(pile, []):
            found.append((card_vocab, card["name"]))
    for power in combat.get("player", {}).get("powers", []):
        found.append((power_vocab, power["name"]))
    for monster in combat.get("monsters", []):
        found.append((monster_id_vocab, monster["id"]))
        found.append((intent_vocab, monster["intent"]))
        for power in monster.get("powers", []):
            found.append((power_vocab, power["name"]))

    return found


@pytest.mark.parametrize("fixture", fixture_names())
def test_every_name_in_a_fixture_is_in_a_vocabulary(fixture):
    unknown = [
        f"{vocabulary.label}: {name!r}"
        for vocabulary, name in names_in(load_payload(fixture))
        if vocabulary.id_of(name) == UNKNOWN_ID
    ]

    assert not unknown, f"{fixture} uses names no vocabulary has: {sorted(set(unknown))}"
