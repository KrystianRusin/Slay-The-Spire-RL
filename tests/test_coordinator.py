"""The Go coordinator, driven from Python through the generated stubs.

Starts the real binary (see tests/stack.py), so this checks the Go service and
the Python side agree on the contract. The Go logic has its own tests in
coordinator/.
"""
import os
import signal
import threading
import time

import grpc
import pytest

pytest.importorskip("rpc.learner_pb2", reason="generate it from rpc/learner.proto first")

from model.policy import TrainingConfig, make_model  # noqa: E402
from rpc import learner_pb2 as pb  # noqa: E402
from rpc import learner_pb2_grpc  # noqa: E402
from rpc.codec import (GRPC_OPTIONS, config_from_proto, config_to_proto,  # noqa: E402
                       decode_state_dict, encode_state_dict, state_dict_digest)
from tests.stack import COORDINATOR_BIN, LONG_POLL_SECONDS, Coordinator  # noqa: E402

CONFIG = TrainingConfig(n_steps=16, policy_layers=(32,), value_layers=(16,))

if not os.path.exists(COORDINATOR_BIN):
    pytest.skip(f"no coordinator binary at {COORDINATOR_BIN}", allow_module_level=True)


@pytest.fixture
def coordinator():
    """A running coordinator on a free port: (Coordinator, actors stub, trainer stub)."""
    running = Coordinator()
    channel = grpc.insecure_channel(running.address, options=GRPC_OPTIONS)
    yield running, learner_pb2_grpc.ActorsStub(channel), learner_pb2_grpc.TrainerStub(channel)
    channel.close()
    running.kill()


def weights(version, run_id, model=None):
    model = model or make_model(CONFIG)
    return pb.Weights(version=version, run_id=run_id, config=config_to_proto(CONFIG),
                      state_dict=encode_state_dict(model.policy.state_dict()))


def fetch(actors, min_version=0, run_id="", timeout=10):
    return actors.GetWeights(pb.GetWeightsRequest(actor_id="actor-a", min_version=min_version, run_id=run_id),
                             timeout=timeout)


def test_an_actor_gets_exactly_what_the_learner_published(coordinator):
    _, actors, trainer = coordinator
    model = make_model(CONFIG)

    ack = trainer.PublishWeights(weights(3, "run-a", model))
    reply = fetch(actors)

    assert ack.replaced
    assert (reply.version, reply.run_id) == (3, "run-a")
    assert config_from_proto(reply.config) == CONFIG
    assert state_dict_digest(decode_state_dict(reply.state_dict)) == state_dict_digest(model.policy.state_dict())


def test_a_waiting_actor_gets_the_next_version_as_soon_as_it_is_published(coordinator):
    _, actors, trainer = coordinator
    trainer.PublishWeights(weights(0, "run-a"))
    result = {}
    waiter = threading.Thread(target=lambda: result.update(reply=fetch(actors, min_version=1, run_id="run-a")))
    waiter.start()
    time.sleep(0.2)

    trainer.PublishWeights(weights(1, "run-a"))
    waiter.join(timeout=5)

    assert result["reply"].version == 1 and result["reply"].HasField("config")


def test_after_the_long_poll_the_reply_has_the_version_but_no_weights(coordinator):
    _, actors, trainer = coordinator
    trainer.PublishWeights(weights(0, "run-a"))

    started = time.monotonic()
    reply = fetch(actors, min_version=1, run_id="run-a")

    assert time.monotonic() - started >= LONG_POLL_SECONDS * 0.9
    assert reply.version == 0 and not reply.HasField("config") and not reply.state_dict


def test_an_actor_from_an_earlier_learner_run_is_answered_at_once(coordinator):
    _, actors, trainer = coordinator
    trainer.PublishWeights(weights(5, "run-b"))

    reply = fetch(actors, min_version=6, run_id="run-a", timeout=LONG_POLL_SECONDS / 2)

    assert (reply.version, reply.run_id) == (5, "run-b")


def test_republishing_the_same_version_is_a_no_op(coordinator):
    _, _, trainer = coordinator

    assert trainer.PublishWeights(weights(1, "run-a")).replaced
    assert not trainer.PublishWeights(weights(1, "run-a")).replaced


def test_weights_without_a_config_are_refused(coordinator):
    _, _, trainer = coordinator

    with pytest.raises(grpc.RpcError) as error:
        trainer.PublishWeights(pb.Weights(version=1, run_id="run-a", state_dict=b"x"))

    assert error.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_it_reports_which_version_it_holds(coordinator):
    _, _, trainer = coordinator
    empty = trainer.CurrentVersion(pb.CurrentVersionRequest())

    trainer.PublishWeights(weights(4, "run-a"))
    held = trainer.CurrentVersion(pb.CurrentVersionRequest())

    assert (empty.version, empty.run_id) == (0, "")
    assert (held.version, held.run_id) == (4, "run-a")


def test_it_stops_cleanly_on_sigterm(coordinator):
    running, _, _ = coordinator

    running.process.send_signal(signal.SIGTERM)

    assert running.wait(timeout=5) == 0
    assert any("stopping" in line for line in running.log)
