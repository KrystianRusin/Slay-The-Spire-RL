"""The STS2 environment against real host payloads (tests/fixtures/sts2).

Each fixture is the first payload of one decision type that the headless host
sent during random-bot runs, so these check the encoder and the legal-action
table on the shapes the game actually produces.
"""
import copy

import numpy as np
import pytest

from sts2.env import (CONFIRM, END_TURN, GROUPS, N_ACTIONS, OPTION_BASE, PLAY_BASE, SKIP, TARGETS,
                      Sts2Env)
from sts2.vocab import VOCABS
from tests.conftest import load_sts2, sts2_fixture_names


def env_on(payload, character="Ironclad"):
    env = Sts2Env(character)
    env.update_game_state(payload)
    return env


@pytest.mark.parametrize("name", sts2_fixture_names())
def test_every_payload_encodes_inside_the_declared_space(name):
    env = env_on(load_sts2(name))
    obs = env.observe()
    assert set(obs) == set(env.observation_space.spaces)
    for key, space in env.observation_space.spaces.items():
        assert obs[key].shape == space.shape, key
        assert space.contains(obs[key]), key


@pytest.mark.parametrize("name", sts2_fixture_names())
def test_every_payload_has_a_legal_action_and_a_command_for_it(name):
    env = env_on(load_sts2(name))
    commands = env.commands()
    mask = env.action_mask()
    assert mask.shape == (N_ACTIONS,) and mask.any()
    assert set(np.flatnonzero(mask)) == set(commands)
    for command in commands.values():
        assert command is None or command["cmd"] == "action"


def test_combat_offers_each_playable_card_at_each_live_enemy_or_untargeted():
    payload = load_sts2("combat_play")
    env = env_on(payload)
    commands = env.commands()
    assert END_TURN in commands
    enemies = [e for e in payload["enemies"] if e["hp"] > 0]
    for h, card in enumerate(payload["hand"]):
        slots = {a for a in commands if PLAY_BASE + h * TARGETS <= a < PLAY_BASE + (h + 1) * TARGETS}
        if not card["can_play"]:
            assert not slots
        elif card["target_type"] == "AnyEnemy":
            assert len(slots) == len(enemies)
            for a in slots:
                target = a - PLAY_BASE - h * TARGETS - 1
                assert commands[a]["args"] == {"card_index": card["index"],
                                               "target_index": enemies[target]["index"]}
        else:
            assert slots == {PLAY_BASE + h * TARGETS}


def test_an_unplayable_card_is_not_offered():
    payload = copy.deepcopy(load_sts2("combat_play"))
    payload["hand"][0]["can_play"] = False
    commands = env_on(payload).commands()
    assert not any(PLAY_BASE <= a < PLAY_BASE + TARGETS for a in commands)


def test_names_in_the_payloads_are_in_the_vocabularies():
    combat = load_sts2("combat_play")
    assert all(card["name"] in VOCABS["card"] for card in combat["hand"])
    assert all(enemy["name"] in VOCABS["monster"] for enemy in combat["enemies"])
    assert all(relic["name"] in VOCABS["relic"] for relic in combat["player"]["relics"])
    obs = env_on(combat).observe()
    assert (obs["hand_ids"][: len(combat["hand"]), 0] > 0).all()


def test_the_character_is_part_of_the_observation():
    payload = load_sts2("map_select")
    a = env_on(payload, "Ironclad").observe()["global_ids"][0, 0]
    b = env_on(payload, "Regent").observe()["global_ids"][0, 0]
    assert a != b and a > 0 and b > 0


def test_a_shop_offers_only_what_you_can_afford_and_leaving():
    payload = copy.deepcopy(load_sts2("shop"))
    payload["player"]["gold"] = 0
    commands = env_on(payload).commands()
    assert set(commands) == {SKIP}
    assert commands[SKIP]["action"] == "leave_room"


def test_a_multi_card_selection_is_picked_one_card_at_a_time():
    payload = copy.deepcopy(load_sts2("card_select"))
    payload["min_select"], payload["max_select"] = 2, 2
    env = env_on(payload)
    first = env.commands()
    assert all(command is None for a, command in first.items() if a >= OPTION_BASE)
    assert CONFIRM not in first

    env.apply_internal(OPTION_BASE + 1)
    assert env.observe()["options_num"][1][GROUPS["options"].num.index("selected")] == 1.0
    second = env.commands()
    assert OPTION_BASE + 1 not in second  # already picked
    assert second[OPTION_BASE + 3]["args"] == {"indices": "1,3"}  # completes the selection


def test_a_single_card_selection_is_sent_at_once():
    commands = env_on(load_sts2("card_select")).commands()
    assert commands[OPTION_BASE]["args"] == {"indices": "0"}


def test_a_refused_action_is_not_offered_again_until_the_next_payload():
    payload = load_sts2("map_select")
    env = env_on(payload)
    env.refused.add(OPTION_BASE)
    assert OPTION_BASE not in env.commands()
    env.update_game_state(payload)
    assert OPTION_BASE in env.commands()


def test_climbing_a_floor_pays_and_losing_hp_costs():
    env = Sts2Env()
    before = load_sts2("map_select")
    after = copy.deepcopy(before)
    after["context"]["floor"] += 1
    assert env.calculate_reward(before, after) == pytest.approx(1.0 - 0.005)

    hurt = copy.deepcopy(before)
    hurt["player"]["hp"] -= 10
    assert env.calculate_reward(before, hurt) == pytest.approx(-3.0 * 10 / hurt["player"]["max_hp"] - 0.005)


def test_damaging_enemies_pays_their_share_of_hp():
    env = Sts2Env()
    before = load_sts2("combat_play")
    after = copy.deepcopy(before)
    enemy = after["enemies"][0]
    enemy["hp"] -= 10
    total = sum(e["max_hp"] for e in before["enemies"])
    assert env.calculate_reward(before, after) == pytest.approx(10 / total - 0.005)


def test_winning_a_fight_and_the_run_pay_and_dying_costs():
    env = Sts2Env()
    combat = load_sts2("combat_play")
    reward = load_sts2("card_reward")
    reward = copy.deepcopy(reward)
    reward["player"] = combat["player"]
    reward["context"] = combat["context"]
    assert env.calculate_reward(combat, reward) == pytest.approx(1.0 - 0.005)

    over = copy.deepcopy(load_sts2("game_over"))
    over["player"] = combat["player"]
    over["context"] = combat["context"]
    over["act"], over["floor"] = combat["context"]["act"], combat["context"]["floor"]
    over["victory"] = False
    assert env.calculate_reward(combat, over) == pytest.approx(-1.0 - 0.005)
    over["victory"] = True
    assert env.calculate_reward(combat, over) == pytest.approx(10.0 - 0.005)
    assert Sts2Env.is_done(over)
