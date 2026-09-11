"""Guard the fixture set itself.

The fixtures are the only stand-in for a running copy of Slay the Spire, so a
fixture that quietly loses a field would weaken every test built on it without
failing anything.
"""

from collections import Counter

from tests.conftest import SCREEN_COVERAGE, fixture_names, load_payload

REQUIRED_SCREENS = {
    "NONE",  # combat, with monsters present
    "CARD_REWARD",
    "MAP",
    "SHOP_SCREEN",
    "REST",
    "GAME_OVER",
}


def screens_on_disk():
    """The screen type each committed fixture actually declares."""
    screens = {}
    for name in fixture_names():
        game_state = load_payload(name).get("game_state")
        screens[name] = game_state["screen_type"] if game_state else None
    return screens


def test_every_fixture_is_covered_by_the_screen_map():
    assert set(fixture_names()) == set(SCREEN_COVERAGE), (
        "a fixture was added or removed without updating SCREEN_COVERAGE in conftest"
    )


def test_the_required_screens_all_have_a_fixture():
    missing = REQUIRED_SCREENS - set(screens_on_disk().values())
    assert not missing, f"no fixture covers {', '.join(sorted(missing))}"


def test_each_fixture_declares_the_screen_it_claims(named_payload):
    name, payload = named_payload
    expected = SCREEN_COVERAGE[name]

    if expected is None:
        assert "game_state" not in payload, f"{name} should be a pre-run payload"
        return

    assert payload["game_state"]["screen_type"] == expected


def test_each_fixture_looks_like_a_communication_mod_payload(named_payload):
    name, payload = named_payload

    assert isinstance(payload.get("available_commands"), list)
    assert payload["available_commands"], f"{name} offers the agent no commands"
    assert isinstance(payload.get("in_game"), bool)

    if SCREEN_COVERAGE[name] is None:
        return

    game_state = payload["game_state"]
    for field in ("deck", "relics", "potions", "map", "screen_state",
                  "current_hp", "max_hp", "gold", "floor", "ascension_level"):
        assert field in game_state, f"{name} is missing game_state.{field}"

    assert game_state["deck"], f"{name} has an empty deck"
    assert game_state["map"], f"{name} has an empty map"


def test_the_map_fixtures_are_the_size_of_a_real_act(named_payload):
    """A whole act, not a stub: fifteen rows, and rows past the declared high.

    The map component declares a high of 10 while real rows reach 14, so a
    fixture with a short map would hide a bound ticket 05 has to fix.
    """
    name, payload = named_payload
    if SCREEN_COVERAGE[name] is None:
        return

    nodes = payload["game_state"]["map"]
    assert len(nodes) > 40, f"{name} has only {len(nodes)} map nodes"
    assert max(node["y"] for node in nodes) == 14


def test_the_combat_fixture_has_live_monsters_and_a_hand():
    combat = load_payload("combat_with_monsters")["game_state"]["combat_state"]

    assert [m for m in combat["monsters"] if not m["is_gone"]]
    assert combat["hand"]
    assert any(card["is_playable"] for card in combat["hand"])


def test_the_combat_piles_add_up_to_the_deck():
    """Hand, draw and discard partition the deck, as they do in a real run."""
    game_state = load_payload("combat_with_monsters")["game_state"]
    combat = game_state["combat_state"]

    in_play = combat["hand"] + combat["draw_pile"] + combat["discard_pile"]
    assert Counter(card["name"] for card in in_play) == Counter(
        card["name"] for card in game_state["deck"]
    )


def test_the_sentinel_costs_are_represented():
    """Ticket 05 has to re-encode these, so they must appear in live data.

    An X-cost card reports -1 and an unplayable curse reports -2. Both sit
    below the declared lower bound of the hand and deck components. They are
    in the hand rather than only in the deck because the deck observation is
    dead until ticket 04.
    """
    game_state = load_payload("combat_with_monsters")["game_state"]
    hand_costs = {card["cost"] for card in game_state["combat_state"]["hand"]}

    assert -1 in hand_costs, "no X-cost card in hand"
    assert -2 in hand_costs, "no unplayable card in hand"


def test_the_combat_pair_represents_a_combat_ending():
    """Ticket 04 needs a before/after pair for the combat-completion reward."""
    before = load_payload("combat_with_monsters")["game_state"]
    after = load_payload("combat_reward")["game_state"]

    assert before["screen_type"] == "NONE"
    assert after["screen_type"] == "COMBAT_REWARD"
    assert before["floor"] == after["floor"], "the pair must be the same floor"
