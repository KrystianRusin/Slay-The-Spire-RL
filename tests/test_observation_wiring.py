"""Each observation component is fed the level of the game state it expects.

The payload arriving over the socket wraps the game state in an envelope. A
builder handed the envelope where it wants the inner state finds none of the
keys it reads and silently produces zeros, so these tests pin the nesting level
by asserting on the data that has to come through.
"""

import copy

import numpy as np
import pytest

from observations.screen_observations import (MAX_SCREEN_OBSERVATION_SIZE,
                                              SCREEN_HANDLERS,
                                              get_screen_observation)
from slay_the_spire_env import SlayTheSpireEnv
from util.vocabularies import (card_vocab, map_symbol_vocab, rest_vocab,
                               reward_type_vocab, screen_type_vocab)

from tests.conftest import SCREEN_COVERAGE, load_payload

# Every component, and a fixture whose game state has to make it non-zero.
COMPONENT_COVERAGE = {
    "player": "combat_with_monsters",
    "hand": "combat_with_monsters",
    "monsters": "combat_with_monsters",
    "deck": "combat_with_monsters",
    "potion": "combat_with_monsters",
    "map": "map_screen",
    "relics": "combat_with_monsters",
    "screen": "shop_screen",
    "extra_info": "combat_with_monsters",
}

# The screen component always opens with its screen-type token, so a non-zero
# check there is only a guard if it looks past that token.
FIRST_DATA_INDEX = {"screen": 1}

# Fixtures whose screen type has a handler of its own, rather than the default.
DISPATCHED_SCREENS = sorted(
    name for name, screen_type in SCREEN_COVERAGE.items() if screen_type in SCREEN_HANDLERS
)

COMBAT_ENDED_REWARD = 40


def observation_for(name):
    payload = load_payload(name)
    return SlayTheSpireEnv(payload).flatten_observation(payload)


def screen_observation_for(name):
    return observation_for(name)["screen"]


@pytest.mark.parametrize("component,fixture", sorted(COMPONENT_COVERAGE.items()))
def test_every_component_carries_data_for_at_least_one_fixture(component, fixture):
    values = np.asarray(observation_for(fixture)[component])[FIRST_DATA_INDEX.get(component, 0):]

    assert np.any(values), f"{component} is all zeros for {fixture}"


def test_the_deck_observation_holds_a_row_per_card():
    deck = load_payload("combat_with_monsters")["game_state"]["deck"]
    name_column = 2  # exhausts, cost, name, type, rarity, ...

    rows = observation_for("combat_with_monsters")["deck"]

    assert len(deck) > 0, "the fixture is meant to have a deck"
    assert int(np.count_nonzero(rows.any(axis=1))) == len(deck)
    assert rows[0][name_column] == card_vocab.id_of(deck[0]["name"])


@pytest.mark.parametrize("fixture", DISPATCHED_SCREENS)
def test_a_screen_with_a_handler_does_not_fall_through_to_the_default(fixture):
    observation = screen_observation_for(fixture)

    assert observation[0] == screen_type_vocab.id_of(SCREEN_COVERAGE[fixture])
    assert np.any(observation[1:]), "nothing past the screen-type token: the default handler ran"


def test_a_screen_without_a_handler_gets_the_default():
    observation = screen_observation_for("game_over")

    assert observation[0] == screen_type_vocab.id_of("GAME_OVER")
    assert not np.any(observation[1:])


def test_the_card_reward_screen_carries_the_cards_on_offer():
    cards = load_payload("card_reward")["game_state"]["screen_state"]["cards"]
    skip_available = 2
    first_card_name = 6  # token, bowl, skip, exhausts, is_playable, cost, name

    observation = screen_observation_for("card_reward")

    assert observation[skip_available] == 1.0
    assert observation[first_card_name] == card_vocab.id_of(cards[0]["name"])


def test_the_combat_reward_screen_carries_the_reward_types():
    rewards = load_payload("combat_reward")["game_state"]["screen_state"]["rewards"]
    first_reward = 1  # token, then one entry per reward

    observation = screen_observation_for("combat_reward")

    assert observation[first_reward] == reward_type_vocab.id_of(rewards[0]["reward_type"])


def test_the_map_screen_carries_the_current_node():
    node = load_payload("map_screen")["game_state"]["screen_state"]["current_node"]
    current_node = slice(2, 5)  # token, first node chosen, symbol, x, y

    observation = screen_observation_for("map_screen")

    assert list(observation[current_node]) == [
        map_symbol_vocab.id_of(node["symbol"]), node["x"], node["y"],
    ]


def test_the_rest_screen_carries_the_options_on_offer():
    options = load_payload("rest_site")["game_state"]["screen_state"]["rest_options"]
    first_rest_option = 2  # token, has rested, then one entry per option

    observation = screen_observation_for("rest_site")

    assert observation[first_rest_option] == rest_vocab.id_of(options[0])


def test_the_shop_screen_carries_the_purge_price():
    screen_state = load_payload("shop_screen")["game_state"]["screen_state"]
    purge_cost = 1  # token, purge cost, purge available, ...

    observation = screen_observation_for("shop_screen")

    assert observation[purge_cost] == screen_state["purge_cost"]


@pytest.mark.parametrize("screen_type", sorted(SCREEN_HANDLERS))
def test_every_handler_returns_an_observation_of_the_declared_width(screen_type):
    """A screen with no fixture still has to survive its handler."""
    observation = get_screen_observation({"screen_type": screen_type, "screen_state": {}})

    assert observation.shape == (MAX_SCREEN_OBSERVATION_SIZE,)
    assert observation[0] == screen_type_vocab.id_of(screen_type)


@pytest.mark.xfail(
    strict=True,
    reason="HAND_SELECT and GRID build more features than the declared screen "
           "width holds, so get_screen_observation truncates them. Widening "
           "the component is a change to the observation space, not to wiring.",
)
def test_no_handler_builds_more_features_than_the_declared_width():
    """Anything a handler builds past the declared width is silently cut."""
    built = {
        screen_type: len(handler({}, screen_type_vocab.id_of(screen_type)))
        for screen_type, handler in sorted(SCREEN_HANDLERS.items())
    }
    overflowing = {k: v for k, v in built.items() if v > MAX_SCREEN_OBSERVATION_SIZE}

    assert not overflowing, (
        f"declared width {MAX_SCREEN_OBSERVATION_SIZE}, truncated: {overflowing}"
    )


def reward_for(previous, current, action="END"):
    env = SlayTheSpireEnv(current)
    env.previous_state = previous
    env.state = current
    env.previous_action = env.action_ids[action]
    return env.calculate_reward()


def test_finishing_a_combat_is_rewarded():
    """Combat over: the screen type goes from NONE to the reward screen."""
    before = load_payload("combat_with_monsters")
    after = load_payload("combat_reward")

    still_in_combat = copy.deepcopy(after)
    still_in_combat["game_state"]["screen_type"] = "NONE"

    ended = reward_for(before, after)
    did_not_end = reward_for(before, still_in_combat)

    assert ended - did_not_end == pytest.approx(COMBAT_ENDED_REWARD)


def test_the_reward_needs_no_previous_state():
    env = SlayTheSpireEnv(load_payload("combat_with_monsters"))
    env.previous_action = env.action_ids["END"]

    assert env.calculate_reward() == 0
