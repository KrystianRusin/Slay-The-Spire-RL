"""Training rides out the learner or the coordinator going down and coming back."""
import threading
import time

import pytest

pytest.importorskip("rpc.learner_pb2", reason="generate it from rpc/learner.proto first")

from model.policy import TrainingConfig  # noqa: E402
from rpc import client as client_module  # noqa: E402
from rpc.client import LearnerRestartedWithNewConfig  # noqa: E402
from rpc.codec import state_dict_digest  # noqa: E402
from tests.rollouts import fill, make_buffer  # noqa: E402
from tests.stack import HEARTBEAT_SECONDS, eventually, stack, wait_until_published  # noqa: E402,F401

CONFIG = TrainingConfig(n_steps=16, policy_layers=(32,), value_layers=(16,))


@pytest.fixture(autouse=True)
def quick_retries(monkeypatch):
    monkeypatch.setattr(client_module, "RETRY_FIRST_SECONDS", 0.05)
    monkeypatch.setattr(client_module, "RETRY_MAX_SECONDS", 0.2)
    monkeypatch.setattr(client_module, "CALL_TIMEOUT", 2)


@pytest.fixture
def rollout(observation_space):
    return fill(make_buffer(observation_space, size=16))


def in_background(function):
    """Run function in a thread; returns a dict that gets its 'result' or 'error'."""
    outcome = {}

    def run():
        try:
            outcome["result"] = function()
        except Exception as e:  # noqa: BLE001 - handed to the test
            outcome["error"] = e

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    outcome["thread"] = thread
    return outcome


# The learner restarting.

def test_a_call_made_while_the_learner_is_down_waits_for_it_to_come_back(stack, rollout):
    first, port = stack.start_learner(CONFIG)
    client = stack.client("actor-a", port)
    client.get_weights()
    stack.stop_learner(first)

    submit = in_background(lambda: client.put(rollout))
    time.sleep(3)  # longer than CALL_TIMEOUT, so at least one attempt has failed
    assert submit["thread"].is_alive()

    second, _ = stack.start_learner(CONFIG, port=port)  # resumes at v0, as a checkpointed learner would
    submit["thread"].join(timeout=10)
    assert "error" not in submit, submit.get("error")
    assert submit["result"].accepted
    assert second.accepted == 1


def test_an_actor_waiting_for_a_version_lost_in_a_restart_takes_the_new_learners(stack, rollout):
    first, port = stack.start_learner(CONFIG, num_actors=2)  # one rollout is in, the round never completes
    client = stack.client("actor-a", port)
    client.get_weights()
    assert client.put(rollout).accepted
    waiting = in_background(client.next_weights)  # for v1, which this learner will never publish
    time.sleep(0.3)

    stack.stop_learner(first)
    second, _ = stack.start_learner(CONFIG, num_actors=2, port=port)
    waiting["thread"].join(timeout=10)

    assert not waiting["thread"].is_alive(), "still waiting for a version that died with the learner"
    assert "error" not in waiting, waiting.get("error")
    assert client.run_id == second.run_id != first.run_id
    assert client.version == 0
    assert state_dict_digest(waiting["result"]) == state_dict_digest(second.policy.state_dict())


def test_a_learner_back_with_a_different_network_stops_the_actor(stack):
    first, port = stack.start_learner(CONFIG)
    client = stack.client("actor-a", port)
    client.get_weights()
    stack.stop_learner(first)
    second, _ = stack.start_learner(TrainingConfig(n_steps=16, policy_layers=(64,), value_layers=(16,)), port=port)
    wait_until_published(second)

    with pytest.raises(LearnerRestartedWithNewConfig, match="policy_layers"):
        client.get_weights()


def test_a_rollout_the_new_learner_accepted_still_waits_for_the_next_version(stack, rollout):
    first, port = stack.start_learner(CONFIG, num_actors=2)
    client = stack.client("actor-a", port)
    client.get_weights()
    stack.stop_learner(first)
    second, _ = stack.start_learner(CONFIG, num_actors=2, port=port)
    wait_until_published(second)
    assert client.put(rollout).accepted  # into the new round

    waiting = in_background(client.next_weights)
    waiting["thread"].join(timeout=2)

    # Its rollout counts towards v1, so it waits for v1 rather than playing v0 again.
    assert waiting["thread"].is_alive()
    assert client.run_id == second.run_id


# The coordinator restarting. It holds the weights only in memory, so it comes
# back with none, and the learner's heartbeat has to notice and publish again.

def test_a_restarted_coordinator_gets_the_weights_back_within_a_few_heartbeats(stack):
    service, port = stack.start_learner(CONFIG)
    wait_until_published(service)
    stack.coordinator.kill()

    started = time.monotonic()
    stack.start_coordinator()
    client = stack.client("actor-a", port)
    version, _, state_dict = client.get_weights()

    assert time.monotonic() - started < 50 * HEARTBEAT_SECONDS
    assert (version, client.run_id) == (0, service.run_id)
    assert state_dict_digest(state_dict) == state_dict_digest(service.policy.state_dict())
    eventually(lambda: service.publisher.publishes == 2, "expected exactly one re-publish")


def test_a_learner_started_before_the_coordinator_publishes_once_it_is_up(stack):
    stack.start_coordinator()
    stack.coordinator.kill()  # now the stack knows a port, but nothing is listening on it
    service, port = stack.start_learner(CONFIG)
    time.sleep(10 * HEARTBEAT_SECONDS)  # several checks fail meanwhile
    assert service.publisher.publishes == 0

    stack.start_coordinator()

    wait_until_published(service)
    assert stack.client("actor-a", port).get_weights()[0] == 0


def test_an_actor_waiting_through_a_coordinator_restart_gets_the_next_version(stack, rollout):
    service, port = stack.start_learner(CONFIG, num_actors=2)
    service.start_training()
    first, second = stack.client("actor-a", port), stack.client("actor-b", port)
    first.get_weights()
    second.get_weights()
    assert first.put(rollout).accepted
    waiting = in_background(first.next_weights)
    time.sleep(0.3)

    stack.coordinator.kill()  # the waiting call fails; the client retries until it's back
    time.sleep(0.5)
    stack.start_coordinator()
    assert second.put(rollout).accepted  # completes the round: the learner trains and publishes v1
    waiting["thread"].join(timeout=30)

    assert "error" not in waiting, waiting.get("error")
    assert first.version == 1
    assert state_dict_digest(waiting["result"]) == state_dict_digest(service.policy.state_dict())


def test_the_heartbeat_doesnt_resend_weights_the_coordinator_already_has(stack):
    service, _ = stack.start_learner(CONFIG)
    wait_until_published(service)

    time.sleep(10 * HEARTBEAT_SECONDS)  # ten checks, all finding v0 there

    assert service.publisher.publishes == 1
    assert sum("published weights" in line for line in stack.coordinator.log) == 1
