"""Actors ride out the learner going down and coming back."""
import threading
import time

import pytest

pytest.importorskip("rpc.learner_pb2", reason="generate it from rpc/learner.proto first")

from learner_server import LearnerService, serve  # noqa: E402
from model.policy import TrainingConfig  # noqa: E402
from rpc import client as client_module  # noqa: E402
from rpc import learner_pb2 as pb  # noqa: E402
from rpc.client import LearnerClient, LearnerRestartedWithNewConfig  # noqa: E402
from rpc.codec import state_dict_digest  # noqa: E402
from tests.rollouts import fill, make_buffer  # noqa: E402

CONFIG = TrainingConfig(n_steps=16, policy_layers=(32,), value_layers=(16,))


@pytest.fixture(autouse=True)
def quick_retries(monkeypatch):
    monkeypatch.setattr(client_module, "RETRY_FIRST_SECONDS", 0.05)
    monkeypatch.setattr(client_module, "RETRY_MAX_SECONDS", 0.2)
    monkeypatch.setattr(client_module, "CALL_TIMEOUT", 2)


@pytest.fixture
def learners():
    """start(port=0, config=CONFIG, num_actors=1) runs a learner; stop(server) kills it."""
    servers = []

    def start(port=0, config=CONFIG, num_actors=1):
        service = LearnerService(config, num_actors)
        service.long_poll_seconds = 0.5
        server, bound = serve(service, port)
        servers.append(server)
        return service, server, bound

    yield start
    for server in servers:
        server.stop(grace=None)


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


def test_a_call_made_while_the_learner_is_down_waits_for_it_to_come_back(learners, observation_space):
    first, server, port = learners()
    client = LearnerClient(f"localhost:{port}", "actor-a")
    client.get_weights()
    server.stop(grace=None).wait()

    submit = in_background(lambda: client.put(fill(make_buffer(observation_space, size=16))))
    time.sleep(3)  # longer than CALL_TIMEOUT, so at least one attempt has failed
    assert submit["thread"].is_alive()

    second, _, _ = learners(port)  # resumes at v0, as a checkpointed learner would
    submit["thread"].join(timeout=10)
    assert "error" not in submit, submit.get("error")
    assert submit["result"].accepted
    assert second.accepted == 1


def test_an_actor_waiting_for_a_version_lost_in_a_restart_takes_the_new_learners(learners, observation_space):
    first, server, port = learners(num_actors=2)  # one rollout is in, the round never completes
    client = LearnerClient(f"localhost:{port}", "actor-a")
    client.get_weights()
    assert client.put(fill(make_buffer(observation_space, size=16))).accepted
    waiting = in_background(client.next_weights)  # for v1, which this learner will never publish
    time.sleep(0.3)

    server.stop(grace=None).wait()
    second, _, _ = learners(port, num_actors=2)
    waiting["thread"].join(timeout=10)

    assert not waiting["thread"].is_alive(), "still waiting for a version that died with the learner"
    assert "error" not in waiting, waiting.get("error")
    assert client.run_id == second.run_id != first.run_id
    assert client.version == 0
    assert state_dict_digest(waiting["result"]) == state_dict_digest(second.policy.state_dict())


def test_the_learner_answers_an_actor_from_an_earlier_run_at_once(learners):
    service, _, _ = learners()
    service.long_poll_seconds = 30

    started = time.monotonic()
    reply = service.GetWeights(pb.GetWeightsRequest(actor_id="a", min_version=7, run_id="old-run"), None)

    assert time.monotonic() - started < 1
    assert reply.version == 0 and reply.state_dict and reply.run_id == service.run_id


def test_the_same_run_still_makes_the_actor_wait(learners):
    service, _, _ = learners()

    reply = service.GetWeights(pb.GetWeightsRequest(actor_id="a", min_version=7, run_id=service.run_id), None)

    assert reply.version == 0 and not reply.state_dict  # not yet


def test_a_learner_back_with_a_different_network_stops_the_actor(learners):
    _, server, port = learners()
    client = LearnerClient(f"localhost:{port}", "actor-a")
    client.get_weights()
    server.stop(grace=None).wait()
    learners(port, config=TrainingConfig(n_steps=16, policy_layers=(64,), value_layers=(16,)))

    with pytest.raises(LearnerRestartedWithNewConfig, match="policy_layers"):
        client.get_weights()


def test_a_rollout_the_new_learner_accepted_still_waits_for_the_next_version(learners, observation_space):
    _, server, port = learners(num_actors=2)
    client = LearnerClient(f"localhost:{port}", "actor-a")
    client.get_weights()
    server.stop(grace=None).wait()
    second, _, _ = learners(port, num_actors=2)
    assert client.put(fill(make_buffer(observation_space, size=16))).accepted  # into the new round

    waiting = in_background(client.next_weights)
    waiting["thread"].join(timeout=2)

    # Its rollout counts towards v1, so it waits for v1 rather than playing v0 again.
    assert waiting["thread"].is_alive()
    assert client.run_id == second.run_id
