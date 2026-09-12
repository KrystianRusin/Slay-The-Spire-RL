"""The action mask: which of the environment's actions are legal in a state."""

import copy
import itertools

import numpy as np
import pytest

from slay_the_spire_env import SlayTheSpireEnv

from tests.conftest import fixture_names, load_payload

START_ACTIONS = {"START IRONCLAD 0", "START SILENT 0"}

# The healthy Block Potion in slot 1 is usable everywhere; the Fire Potion in
# slot 0 needs a live monster to target, so it can only be used in combat.
UNTARGETED_POTION = {"POTION Use 1", "POTION Discard 0", "POTION Discard 1"}

LEGAL_ACTIONS = {
    "main_menu": START_ACTIONS,
    "combat_with_monsters": UNTARGETED_POTION | {
        "POTION Use 0 0", "POTION Use 0 1",
        "PLAY 1 0", "PLAY 1 1",  # Strike
        "PLAY 2 0", "PLAY 2 1",  # Strike
        "PLAY 3",                # Defend
        "PLAY 4 0", "PLAY 4 1",  # Bash
        "PLAY 6",                # Whirlwind
    },
    "combat_reward": UNTARGETED_POTION | {"CHOOSE 0", "CHOOSE 1", "CHOOSE 2", "PROCEED"},
    "card_reward": UNTARGETED_POTION | {"CHOOSE 0", "CHOOSE 1", "CHOOSE 2", "PROCEED"},
    "map_screen": UNTARGETED_POTION | {"CHOOSE 0", "CHOOSE 1", "RETURN"},
    "rest_site": UNTARGETED_POTION | {"CHOOSE 0", "CHOOSE 1"},
    "shop_screen": UNTARGETED_POTION | {f"CHOOSE {i}" for i in range(6)} | {"LEAVE"},
    "game_over": {"PROCEED"},
}

REACHED_BY = ["START IRONCLAD 0", "PLAY 3", "END", "PROCEED", "CHOOSE 0", "RETURN", "LEAVE"]


def env_in(payload):
    """An environment that has just received payload from the game."""
    env = SlayTheSpireEnv({})
    env.update_game_state(payload)
    return env


def legal_actions(env):
    mask = env.get_valid_action_mask(env.state)
    return {action.text for action, legal in zip(env.actions, mask) if legal}


def take(env, action):
    env.step(env.action_ids[action])


def next_turn(payload):
    following = copy.deepcopy(payload)
    following["game_state"]["combat_state"]["turn"] += 1
    return following


def test_every_fixture_has_expected_legal_actions():
    assert set(LEGAL_ACTIONS) == set(fixture_names())


@pytest.mark.parametrize("fixture", sorted(LEGAL_ACTIONS))
def test_the_legal_actions_for_each_fixture(fixture):
    env = env_in(load_payload(fixture))

    assert legal_actions(env) == LEGAL_ACTIONS[fixture]


def test_the_mask_is_one_boolean_per_action():
    env = env_in(load_payload("combat_with_monsters"))

    mask = env.get_valid_action_mask(env.state)

    assert mask.dtype == bool
    assert mask.shape == (env.action_space.n,)


def test_an_empty_state_allows_exactly_the_start_actions():
    env = SlayTheSpireEnv({})

    assert legal_actions(env) == START_ACTIONS


def test_ending_the_turn_is_illegal_before_acting_while_a_card_is_playable():
    env = env_in(load_payload("combat_with_monsters"))

    assert "END" not in legal_actions(env)


def test_ending_the_turn_is_legal_after_playing_a_card():
    combat = load_payload("combat_with_monsters")
    env = env_in(combat)

    take(env, "PLAY 3")
    env.update_game_state(combat)

    assert "END" in legal_actions(env)


def test_ending_the_turn_is_legal_with_no_playable_card():
    combat = copy.deepcopy(load_payload("combat_with_monsters"))
    for card in combat["game_state"]["combat_state"]["hand"]:
        card["is_playable"] = False

    env = env_in(combat)

    assert "END" in legal_actions(env)


def test_the_next_turn_starts_with_no_action_taken():
    combat = load_payload("combat_with_monsters")
    env = env_in(combat)

    take(env, "PLAY 3")
    take(env, "END")
    env.update_game_state(next_turn(combat))

    assert "END" not in legal_actions(env)


def test_a_new_combat_starts_with_no_action_taken():
    combat = load_payload("combat_with_monsters")
    env = env_in(combat)

    take(env, "PLAY 3")
    env.update_game_state(load_payload("combat_reward"))
    take(env, "PROCEED")
    env.update_game_state(combat)

    assert "END" not in legal_actions(env)


@pytest.mark.parametrize("last_action", ["PROCEED", "CHOOSE 0", "RETURN"])
def test_return_is_illegal_straight_after(last_action):
    env = env_in(load_payload("map_screen"))

    take(env, last_action)

    assert "RETURN" not in legal_actions(env)


def test_a_loop_guard_never_leaves_nothing_legal():
    only_return = copy.deepcopy(load_payload("map_screen"))
    only_return["available_commands"] = ["return", "key", "click", "wait", "state"]
    env = env_in(only_return)

    take(env, "PROCEED")

    assert legal_actions(env) == {"RETURN"}


def test_ending_the_turn_is_legal_when_no_play_action_can_play_the_playable_card():
    """The only playable card sits beyond the cards the action table can play."""
    combat = copy.deepcopy(load_payload("combat_with_monsters"))
    hand = combat["game_state"]["combat_state"]["hand"]
    for card in hand:
        card["is_playable"] = False
    hand.extend(copy.deepcopy(hand[0]) for _ in range(10 - len(hand)))
    hand[-1]["is_playable"] = True

    env = env_in(combat)

    assert legal_actions(env) == UNTARGETED_POTION | {"POTION Use 0 0", "POTION Use 0 1", "END"}


@pytest.mark.parametrize(
    "fixture,last_action", list(itertools.product(fixture_names(), REACHED_BY))
)
def test_no_fixture_state_has_nothing_legal(fixture, last_action):
    payload = load_payload(fixture)
    env = env_in(payload)
    take(env, last_action)
    env.update_game_state(payload)

    assert np.any(env.get_valid_action_mask(env.state))
