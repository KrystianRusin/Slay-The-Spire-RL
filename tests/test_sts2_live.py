"""The real-game adapter (sts2/live.py): mod states in the shapes its API
documents become payloads the trained environment can encode, and the
policy's commands become the mod's actions."""
import pytest

from sts2.env import END_TURN, OPTION_BASE, PLAY_BASE, SKIP, TARGETS, Sts2Env
from sts2.live import LiveGame

PLAYER = {
    "character": "The Ironclad", "hp": 70, "max_hp": 80, "block": 0, "gold": 120,
    "relics": [{"id": "BURNING_BLOOD", "name": "Burning Blood"}],
    "potions": [{"id": "FIRE_POTION", "name": "Fire Potion", "slot": 0, "target_type": "AnyEnemy"}],
    "max_potion_slots": 3, "status": [],
}


def combat_state():
    player = dict(PLAYER, energy=3, max_energy=3, draw_pile_count=5, discard_pile_count=0,
                  hand=[{"index": 0, "name": "Strike", "type": "Attack", "cost": "1", "target_type": "AnyEnemy",
                         "can_play": True, "description": "Deal 6 damage.", "is_upgraded": False},
                        {"index": 1, "name": "Defend", "type": "Skill", "cost": "1", "target_type": "Self",
                         "can_play": True, "description": "Gain 5 Block.", "is_upgraded": False}],
                  draw_pile=[{"name": "Bash", "cost": "2"}], discard_pile=[], exhaust_pile=[])
    return {"state_type": "monster", "run": {"act": 1, "floor": 2, "ascension": 0}, "player": player,
            "battle": {"round": 1, "turn": "player", "is_play_phase": True,
                       "enemies": [{"entity_id": "JAW_WORM_0", "name": "Jaw Worm", "hp": 40, "max_hp": 44,
                                    "block": 0, "status": [{"name": "Strength", "amount": 3}],
                                    "intents": [{"type": "Attack", "label": "5x2"}]}]}}


def encode(game, mod_state):
    env = Sts2Env("Ironclad")
    env.update_game_state(game.to_payload(mod_state))
    obs = env.observe()
    for key, space in env.observation_space.spaces.items():
        assert space.contains(obs[key]), key
    return env


def test_a_combat_screen_encodes_and_targets_by_entity_id():
    game = LiveGame()
    env = encode(game, combat_state())
    commands = env.commands()
    assert END_TURN in commands
    strike = commands[PLAY_BASE + 0 * TARGETS + 1]
    assert game.to_requests(strike) == [{"action": "play_card", "card_index": 0, "target": "JAW_WORM_0"}]
    defend = commands[PLAY_BASE + 1 * TARGETS]
    assert game.to_requests(defend) == [{"action": "play_card", "card_index": 1}]
    assert env.state["hand"][0]["stats"] == {"damage": 6}
    assert env.state["enemies"][0]["intents"][0]["total_damage"] == 10


def test_the_deck_seen_in_combat_is_kept_for_later_screens():
    game = LiveGame()
    game.to_payload(combat_state())
    map_state = {"state_type": "map", "run": {"act": 1, "floor": 2}, "player": PLAYER,
                 "map": {"next_options": [{"index": 0, "col": 2, "row": 3, "type": "RestSite"},
                                          {"index": 1, "col": 4, "row": 3, "type": "Elite"}]}}
    env = encode(game, map_state)
    assert len(env.state["player"]["deck"]) == 3
    second = env.commands()[OPTION_BASE + 1]
    assert game.to_requests(second) == [{"action": "choose_map_node", "index": 1}]


def test_a_shop_maps_each_purchase_to_its_flat_index():
    game = LiveGame()
    shop = {"state_type": "shop", "run": {"act": 1, "floor": 5}, "player": PLAYER,
            "shop": {"items": [
                {"index": 0, "category": "card", "price": 50, "is_stocked": True, "card_name": "Uppercut",
                 "card_type": "Attack", "card_cost": "2", "card_description": "Deal 13 damage."},
                {"index": 1, "category": "relic", "price": 150, "is_stocked": True, "relic_name": "Vajra"},
                {"index": 2, "category": "potion", "price": 50, "is_stocked": True, "potion_name": "Fire Potion"},
                {"index": 3, "category": "card_removal", "price": 75, "is_stocked": True}]}}
    env = encode(game, shop)
    commands = env.commands()
    requests = sorted(game.to_requests(c)[0]["index"] for a, c in commands.items() if a != SKIP)
    assert requests == [0, 2, 3]  # the relic costs more than the 120 gold
    assert game.to_requests(commands[SKIP]) == [{"action": "proceed"}]


def test_a_multi_card_selection_becomes_toggles_then_confirm():
    game = LiveGame()
    select = {"state_type": "card_select", "run": {"act": 1, "floor": 1}, "player": PLAYER,
              "card_select": {"screen_type": "transform", "prompt": "Choose 2 cards to Transform.",
                              "cards": [{"index": i, "name": "Strike", "type": "Attack", "cost": "1"}
                                        for i in range(4)]}}
    env = encode(game, select)
    assert env.commands()[OPTION_BASE] is None  # first pick stays local
    env.apply_internal(OPTION_BASE + 2)
    done = env.commands()[OPTION_BASE + 0]
    assert game.to_requests(done) == [{"action": "select_card", "index": 0}, {"action": "select_card", "index": 2},
                                      {"action": "confirm_selection"}]


@pytest.mark.parametrize("option_id, expected", [("rest", "HEAL"), ("smith", "SMITH")])
def test_rest_options_use_the_training_ids(option_id, expected):
    game = LiveGame()
    rest = {"state_type": "rest_site", "run": {"act": 1, "floor": 7}, "player": PLAYER,
            "rest_site": {"options": [{"index": 0, "id": option_id, "is_enabled": True}]}}
    env = encode(game, rest)
    assert env.options[0].ids[7] > 0
    assert env.state["options"][0]["option_id"] == expected
    assert game.to_requests(env.commands()[OPTION_BASE]) == [{"action": "choose_rest_option", "index": 0}]


def menu_clicks(state, character="Ironclad", game=None):
    """The menu options LiveGame clicks for state (without a game to click on)."""
    game = game or LiveGame(log=lambda message: None)
    clicked = []
    game.post = lambda body: clicked.append(body.get("option")) or {"status": "ok"}
    assert game.handle_automatically(state, character)
    return clicked


def test_menus_only_click_toward_a_new_run():
    assert menu_clicks({"state_type": "menu", "menu_screen": "main",
                        "options": ["singleplayer", "multiplayer", "compendium", "settings", "quit"]}) == ["singleplayer"]
    assert menu_clicks({"state_type": "menu", "menu_screen": "singleplayer",
                        "options": ["standard", "daily", "custom", "back"]}) == ["standard"]
    assert menu_clicks({"state_type": "menu", "menu_screen": "multiplayer",
                        "options": ["host", "back"]}) == ["back"]


def test_a_menu_without_the_way_forward_is_left_alone(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    # The main menu while it animates in: singleplayer isn't clickable yet.
    assert menu_clicks({"state_type": "menu", "menu_screen": "main", "options": ["settings", "quit"]}) == []
    # Settings has no options and the mod can't close it.
    assert menu_clicks({"state_type": "menu", "menu_screen": "settings"}) == []


def test_a_run_in_progress_is_continued():
    assert menu_clicks({"state_type": "menu", "menu_screen": "main",
                        "options": ["continue", "abandon_run", "settings", "quit"]}) == ["continue"]


def test_character_select_picks_the_character_then_embarks():
    game = LiveGame(log=lambda message: None)
    screen = {"state_type": "menu", "menu_screen": "character_select",
              "options": [{"name": "IRONCLAD", "enabled": True}, {"name": "DEFECT", "enabled": True},
                          {"name": "back", "enabled": True}]}
    assert menu_clicks(screen, "Defect", game) == ["defect"]
    assert menu_clicks(screen, "Defect", game) == ["embark"]


def test_pending_timeline_unlocks_are_mentioned_once_and_the_run_still_starts():
    said = []
    game = LiveGame(log=said.append)
    main = {"state_type": "menu", "menu_screen": "main", "options": ["singleplayer", "settings", "quit"],
            "blocked_options": [{"name": "timeline", "enabled": False, "reason": "manual_epoch_reveal_required"}]}
    assert menu_clicks(main, game=game) == ["singleplayer"]
    assert menu_clicks(main, game=game) == ["singleplayer"]
    assert sum("Timeline" in line for line in said) == 1
