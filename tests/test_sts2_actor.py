"""The actor loop against a scripted host: which steps reach the buffer, with
which reward and episode end.

The host answers each command with the next payload, so an action's reward is
only known one reply later. Refused actions must not be recorded at all (the
policy picks again), selection picks are recorded without a host round trip,
and a run that ends without a game over still marks its last step as the end.
"""
import copy

import pytest
import torch as th

from environment import run_env
from environment.run_env import Actor, play_run
from model.custom_rollout_buffer import CustomRolloutBuffer
from model.policy import TrainingConfig, make_model
from sts2.env import END_TURN, OPTION_BASE, SKIP, Sts2Env
from sts2.host import HostError
from tests.conftest import load_sts2


class ScriptedHost:
    """Replies with the given payloads in order, and records what it was sent."""

    def __init__(self, first, replies):
        self.first, self.replies, self.sent = first, list(replies), []

    def start_run(self, character, seed, ascension=0):
        return self.first

    def send(self, command):
        self.sent.append(command)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.fixture
def chosen(monkeypatch):
    """Script the policy: append action indices, in the order to play them."""
    script = []

    def scripted_act(model, obs_tensor, action_mask, deterministic=False):
        action = script.pop(0)
        assert action_mask[action], f"scripted action {action} isn't legal"
        return action, th.tensor([[0.0]]), th.tensor([-0.5])

    monkeypatch.setattr(run_env, "act", scripted_act)
    return script


@pytest.fixture
def actor():
    env = Sts2Env()
    config = TrainingConfig(policy_layers=(16,), value_layers=(16,))
    model = make_model(config, env=env)
    buffer = CustomRolloutBuffer(64, env.observation_space, env.action_space, device="cpu",
                                 gamma=model.gamma, gae_lambda=model.gae_lambda, n_envs=1)
    return Actor(env, model, buffer, lambda: None)


def game_over(victory=False):
    over = copy.deepcopy(load_sts2("game_over"))
    over["victory"] = victory
    return over


def test_each_accepted_action_is_one_step_and_the_last_ends_the_run(actor, chosen):
    combat = load_sts2("combat_play")
    chosen += [END_TURN, END_TURN]
    host = ScriptedHost(combat, [combat, game_over()])
    result = play_run(actor, host, "Ironclad", "seed")
    assert result["outcome"] == "defeat"
    assert actor.buffer.pos == 2
    assert actor.buffer.dones[:2].tolist() == [0.0, 1.0]
    assert [c["action"] for c in host.sent] == ["end_turn", "end_turn"]


def test_a_refused_action_is_not_recorded_and_not_offered_again(actor, chosen):
    map_state = load_sts2("map_select")
    chosen += [OPTION_BASE, OPTION_BASE + 1]
    host = ScriptedHost(map_state, [{"type": "error", "message": "nope"}, game_over()])
    play_run(actor, host, "Ironclad", "seed")
    assert actor.buffer.pos == 1  # only the accepted pick
    assert host.sent[1]["args"] == {"col": map_state["choices"][1]["col"], "row": map_state["choices"][1]["row"]}


def test_a_selection_pick_is_a_step_without_a_host_round_trip(actor, chosen):
    select = copy.deepcopy(load_sts2("card_select"))
    select["min_select"], select["max_select"] = 2, 2
    chosen += [OPTION_BASE, OPTION_BASE + 2]
    host = ScriptedHost(select, [game_over()])
    play_run(actor, host, "Ironclad", "seed")
    assert actor.buffer.pos == 2
    assert host.sent == [{"cmd": "action", "action": "select_cards", "args": {"indices": "0,2"}}]


def test_a_host_crash_mid_run_marks_the_last_recorded_step_as_the_end(actor, chosen):
    combat = load_sts2("combat_play")
    chosen += [END_TURN, END_TURN]
    host = ScriptedHost(combat, [combat, HostError("gone")])
    with pytest.raises(HostError):
        play_run(actor, host, "Ironclad", "seed")
    actor.cut_off()
    assert actor.buffer.pos == 1
    assert actor.buffer.dones[0] == 1.0


def test_rewards_are_scaled_into_the_buffer(actor, chosen):
    actor.reward_scale = 0.5
    shop = load_sts2("shop")
    chosen += [SKIP]
    host = ScriptedHost(shop, [game_over(victory=True)])
    play_run(actor, host, "Ironclad", "seed")
    raw = actor.env.calculate_reward(shop, game_over(victory=True))
    assert actor.buffer.rewards[0] == pytest.approx(raw * 0.5)
