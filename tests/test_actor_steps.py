"""Which action each reward and episode end lands on, as the actor records them.

The game answers each command with the next state, so an action's reward and
whether it ended the run are only known one state later. These tests feed the
Actor scripted states and check the buffer it fills.
"""
import copy
import random

import pytest
import torch as th

from environment import run_env
from environment.run_env import Actor
from model.custom_rollout_buffer import CustomRolloutBuffer
from model.policy import TrainingConfig, make_model
from slay_the_spire_env import SlayTheSpireEnv
from tests.conftest import load_payload

STEP_PENALTY = -0.1
HP_LOSS_PENALTY = 3


class Script(list):
    calls = None


@pytest.fixture
def chosen(monkeypatch):
    """Script the policy: append action texts to the returned list, in the order to play them.

    Its .calls lists every time the policy was asked to act.
    """
    script, calls = Script(), []

    def scripted_act(model, obs_tensor, action_mask):
        calls.append("act")
        env = SlayTheSpireEnv({})
        return env.action_ids[script.pop(0)], th.tensor([[0.0]]), th.tensor([-0.5])

    monkeypatch.setattr(run_env, "act", scripted_act)
    script.calls = calls
    return script


@pytest.fixture
def make_actor():
    def make(size=16, on_full=None, rng=None, reward_scale=1.0):
        env = SlayTheSpireEnv({})
        model = make_model(TrainingConfig(policy_layers=(16,), value_layers=(16,)), env=env)
        buffer = CustomRolloutBuffer(size, env.observation_space, env.action_space, device="cpu",
                                     gamma=model.gamma, gae_lambda=model.gae_lambda, n_envs=1)
        return Actor(env, model, buffer, on_full or (lambda: None), rng=rng, reward_scale=reward_scale)
    return make


def combat():
    return copy.deepcopy(load_payload("combat_with_monsters"))


def monster_hit(state, damage):
    hit = copy.deepcopy(state)
    hit["game_state"]["combat_state"]["monsters"][0]["current_hp"] -= damage
    return hit


def player_hit(state, damage):
    hit = copy.deepcopy(state)
    hit["game_state"]["current_hp"] -= damage
    return hit


def died(state):
    dead = copy.deepcopy(state)
    dead["game_state"]["screen_type"] = "GAME_OVER"
    dead["game_state"]["screen_state"] = {"victory": False}
    dead["game_state"]["current_hp"] = 0
    dead["available_commands"] = ["proceed", "state"]
    return dead


def menu():
    return copy.deepcopy(load_payload("main_menu"))


def feed(actor, states):
    return [actor.on_state(state) for state in states]


def test_the_reward_lands_on_the_action_that_caused_it(make_actor, chosen):
    actor = make_actor()
    start = combat()
    max_hp = start["game_state"]["combat_state"]["monsters"][0]["max_hp"]
    chosen += ["PLAY 1 0", "END", "END"]

    feed(actor, [start, monster_hit(start, 10), monster_hit(start, 10)])

    # PLAY did the damage, END after it changed nothing.
    assert actor.buffer.pos == 2
    assert actor.buffer.rewards[0] == pytest.approx(10 / max_hp * 10 + STEP_PENALTY)
    assert actor.buffer.rewards[1] == pytest.approx(STEP_PENALTY)


def test_the_run_ends_on_the_move_that_lost_it(make_actor, chosen):
    actor = make_actor()
    start = combat()
    hp = start["game_state"]["current_hp"]
    chosen += ["PLAY 1 0", "END"]

    commands = feed(actor, [start, player_hit(start, 5), died(player_hit(start, 5))])

    assert commands[:2] == ["PLAY 1 0", "END"]
    assert commands[2] == "PROCEED"  # sent by the actor, not chosen by the policy
    assert actor.buffer.pos == 2  # the game-over screen isn't a step
    assert actor.buffer.dones.tolist()[:2] == [0.0, 1.0]
    assert actor.buffer.rewards[0] == pytest.approx(-5 * HP_LOSS_PENALTY + STEP_PENALTY)
    assert actor.buffer.rewards[1] == pytest.approx(-(hp - 5) * HP_LOSS_PENALTY + STEP_PENALTY)
    assert actor.ended == (pytest.approx(float(actor.buffer.rewards[:2].sum())), 2)


def test_the_main_menu_starts_a_run_without_recording_a_step(make_actor, chosen):
    actor = make_actor(rng=random.Random(0))
    chosen += ["PLAY 1 0"]

    command = actor.on_state(menu())
    actor.on_state(combat())  # the new run's first state

    assert command in {"START IRONCLAD 0", "START SILENT 0"}
    assert chosen.calls == ["act"]  # the policy only acted in the run
    assert actor.buffer.pos == 0  # and that action is still waiting for its result


def test_both_characters_get_played(make_actor, chosen):
    actor = make_actor(rng=random.Random(0))

    characters = {actor.on_state(menu()) for _ in range(40)}

    assert characters == {"START IRONCLAD 0", "START SILENT 0"}
    assert actor.buffer.pos == 0


def test_a_new_run_starts_clean_after_a_death(make_actor, chosen):
    actor = make_actor()
    start = combat()
    chosen += ["END", "PLAY 1 0", "END"]

    feed(actor, [start, died(start), menu(), start, monster_hit(start, 10)])

    assert actor.buffer.pos == 2
    assert actor.buffer.dones.tolist()[:2] == [1.0, 0.0]
    assert actor.buffer.rewards[1] > 0  # the new run's PLAY, rewarded for its own damage
    assert actor.episode_steps == 1


def test_a_full_rollout_bootstraps_from_the_state_after_its_last_step(make_actor, chosen):
    seen = []
    actor = make_actor(size=2, on_full=lambda: seen.append(actor.buffer.returns.copy()))
    start = combat()
    chosen += ["END", "END", "END"]

    feed(actor, [start, start, start])
    with th.no_grad():
        value_after = actor.model.policy.predict_values(
            run_env.to_tensors(actor.env.flatten_observation(start), "cpu")).item()

    assert len(seen) == 1
    # Recorded values are 0, so the last return is its reward plus the discounted bootstrap.
    assert seen[0][1] == pytest.approx(STEP_PENALTY + actor.model.gamma * value_after, rel=1e-5)


def test_a_rollout_that_fills_on_a_death_doesnt_bootstrap(make_actor, chosen):
    seen = []
    actor = make_actor(size=1, on_full=lambda: seen.append(actor.buffer.returns.copy()))
    start = combat()
    hp = start["game_state"]["current_hp"]
    chosen += ["END"]

    feed(actor, [start, died(start)])

    assert seen[0][0] == pytest.approx(-hp * HP_LOSS_PENALTY + STEP_PENALTY)


def test_the_next_action_is_chosen_after_the_rollout_is_handed_off(make_actor, chosen):
    events = chosen.calls
    actor = make_actor(size=1, on_full=lambda: events.append("handed off"))
    chosen += ["END", "END"]

    feed(actor, [combat(), combat()])

    # So an action is always sampled by the weights its rollout is tagged with.
    assert events == ["act", "handed off", "act"]


def test_rewards_are_recorded_scaled_but_logged_unscaled(make_actor, chosen):
    actor = make_actor(reward_scale=0.1)
    start = combat()
    chosen += ["END", "END"]

    feed(actor, [start, player_hit(start, 5), died(player_hit(start, 5))])

    unscaled = -5 * HP_LOSS_PENALTY + STEP_PENALTY
    assert actor.buffer.rewards[0] == pytest.approx(unscaled * 0.1)
    assert actor.ended[0] == pytest.approx(unscaled - (start["game_state"]["current_hp"] - 5) * HP_LOSS_PENALTY
                                           + STEP_PENALTY)
