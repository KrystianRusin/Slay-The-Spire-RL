"""The actor <-> learner contract: message encoding and a real in-process gRPC server.

Needs the code generated from rpc/learner.proto, which the image build creates.
"""
import time

import numpy as np
import pytest
import torch as th

pytest.importorskip("rpc.learner_pb2", reason="generate it from rpc/learner.proto first")

import learner_server  # noqa: E402
from environment.run_env import act  # noqa: E402
from learner_server import LearnerService, serve  # noqa: E402
from model.policy import TrainingConfig, make_model  # noqa: E402
from rpc import learner_pb2 as pb  # noqa: E402
from rpc.client import LearnerClient  # noqa: E402
from rpc.codec import (BUFFER_FIELDS, config_from_proto, config_to_proto,  # noqa: E402
                       decode_state_dict, encode_state_dict, rollout_from_proto,
                       rollout_to_proto, state_dict_digest)
from tests.rollouts import fill, make_buffer, mask_for  # noqa: E402


@pytest.fixture
def rollout(observation_space):
    return fill(make_buffer(observation_space, size=16))


def test_a_rollout_survives_the_round_trip_exactly(rollout):
    message = pb.Rollout.FromString(rollout_to_proto(rollout, "actor-a", 3).SerializeToString())

    decoded = rollout_from_proto(message)

    assert (decoded["actor_id"], decoded["policy_version"], decoded["n_steps"]) == ("actor-a", 3, 16)
    assert decoded["observations"].keys() == rollout.observations.keys()
    for key, array in rollout.observations.items():
        np.testing.assert_array_equal(decoded["observations"][key], array)
        assert decoded["observations"][key].dtype == array.dtype
    for field, attribute in BUFFER_FIELDS.items():
        np.testing.assert_array_equal(decoded[field], getattr(rollout, attribute))
    np.testing.assert_array_equal(decoded["action_masks"], rollout.action_masks)
    assert decoded["action_masks"].dtype == bool


def test_only_the_filled_steps_are_sent(observation_space):
    buffer = make_buffer(observation_space, size=16)
    filled = fill(make_buffer(observation_space, size=16))
    for key in buffer.observations:
        buffer.observations[key][:5] = filled.observations[key][:5]
    buffer.pos = 5

    decoded = rollout_from_proto(rollout_to_proto(buffer, "actor-a", 0))

    assert decoded["n_steps"] == 5
    assert all(array.shape[0] == 5 for array in decoded["observations"].values())


def test_a_rollout_with_a_short_array_is_refused(rollout):
    message = rollout_to_proto(rollout, "actor-a", 0)
    message.n_steps = 17

    with pytest.raises(ValueError, match="expected 17 steps"):
        rollout_from_proto(message)


def test_weights_survive_the_round_trip_and_the_digest_tracks_them():
    state_dict = make_model(TrainingConfig()).policy.state_dict()

    decoded = decode_state_dict(encode_state_dict(state_dict))

    assert state_dict_digest(decoded) == state_dict_digest(state_dict)
    first = next(iter(decoded))
    decoded[first] = decoded[first] + 1
    assert state_dict_digest(decoded) != state_dict_digest(state_dict)


def test_a_config_survives_the_round_trip_exactly():
    config = TrainingConfig(n_steps=512, gamma=0.9, ent_coef=0.01, policy_layers=(128, 32))

    message = pb.PolicyConfig.FromString(config_to_proto(config).SerializeToString())

    assert config_from_proto(message) == config


# A config unlike the defaults, so the tests notice if an actor ignores what the learner sent.
LEARNER_CONFIG = TrainingConfig(n_steps=256, gamma=0.9, policy_layers=(32,), value_layers=(16, 16))


@pytest.fixture
def start_learner():
    """Starts real learners on free ports. The training thread is up to each test."""
    servers = []

    def start(num_actors=1):
        service = LearnerService(LEARNER_CONFIG, num_actors)
        server, port = serve(service, 0)
        servers.append(server)
        return service, lambda actor_id: LearnerClient(f"localhost:{port}", actor_id)

    yield start
    for server in servers:
        server.stop(grace=None)


@pytest.fixture
def learner(start_learner):
    """A real learner that doesn't train, and a client connected to it."""
    service, connect = start_learner()
    return service, connect("actor-test")


def wait_for_version(service, version, timeout=60):
    deadline = time.monotonic() + timeout
    while service.version < version:
        assert time.monotonic() < deadline, f"learner never reached v{version}"
        time.sleep(0.01)


def test_an_actor_gets_the_learners_exact_weights(learner):
    service, client = learner

    version, config, state_dict = client.get_weights()

    assert version == 0
    assert config == LEARNER_CONFIG
    assert state_dict_digest(state_dict) == state_dict_digest(service.policy.state_dict())


def test_an_actor_builds_the_learners_network_from_the_config_it_was_sent(learner):
    service, client = learner
    _, config, state_dict = client.get_weights()

    actor_model = make_model(config)  # what run_environment does
    actor_model.policy.load_state_dict(state_dict)

    assert state_dict_digest(actor_model.policy.state_dict()) == state_dict_digest(service.policy.state_dict())
    assert (actor_model.n_steps, actor_model.gamma) == (256, 0.9)
    # the default network has different layers, so the learner's weights don't fit it
    with pytest.raises(RuntimeError):
        make_model(TrainingConfig()).policy.load_state_dict(state_dict)


def test_a_rollout_made_with_the_current_version_is_accepted(learner, rollout):
    service, client = learner
    client.get_weights()

    ack = client.put(rollout)

    assert ack.accepted
    assert (service.accepted, service.rejected) == (1, 0)


def test_a_rollout_made_with_another_version_is_rejected(learner, rollout):
    service, client = learner
    client.get_weights()
    client.version = 7  # as if the actor were playing with weights the learner has moved past

    ack = client.put(rollout)

    assert not ack.accepted
    assert "v7" in ack.reason
    assert (service.accepted, service.rejected) == (0, 1)


def test_submitting_before_fetching_weights_is_an_error(learner, rollout):
    _, client = learner

    with pytest.raises(RuntimeError, match="get_weights"):
        client.put(rollout)


def test_the_learner_rejects_a_rollout_without_action_masks(learner, rollout):
    service, client = learner
    client.get_weights()
    message = rollout_to_proto(rollout, "actor-test", 0)
    message.ClearField("action_masks")

    ack = client.stub.SubmitRollout(message)

    assert not ack.accepted
    assert "action masks" in ack.reason
    assert service.accepted == 0


def test_the_learner_rejects_a_malformed_rollout(learner, rollout):
    service, client = learner
    message = rollout_to_proto(rollout, "actor-test", 0)
    message.rewards.CopyFrom(pb.Tensor(shape=[3], dtype="float32", data=th.zeros(3).numpy().tobytes()))

    ack = client.stub.SubmitRollout(message)

    assert not ack.accepted
    assert "malformed" in ack.reason


def test_a_rollout_with_the_wrong_observation_shape_is_rejected(learner, observation_space):
    service, client = learner
    client.get_weights()
    wrong = make_buffer(observation_space, size=16)
    key = next(iter(wrong.observations))
    wrong.observations[key] = np.zeros((16, 3), dtype=np.float32)
    wrong.pos = 16

    ack = client.put(wrong)

    assert not ack.accepted
    assert "malformed" in ack.reason
    assert service.accepted == 0


def test_a_full_round_trains_and_publishes_the_next_version(start_learner, rollout):
    service, connect = start_learner()
    service.start_training()
    client = connect("actor-a")
    client.get_weights()
    before = state_dict_digest(service.policy.state_dict())

    client.put(rollout)
    wait_for_version(service, 1)

    version, _, state_dict = client.get_weights()
    assert version == 1
    assert state_dict_digest(state_dict) != before
    assert service.last_stats.steps == 16


def test_the_learner_waits_for_every_actor_and_trains_on_all_their_steps(start_learner, rollout):
    service, connect = start_learner(num_actors=2)
    service.start_training()
    first, second = connect("actor-a"), connect("actor-b")
    first.get_weights()
    second.get_weights()

    first.put(rollout)
    assert service.version == 0
    second.put(rollout)
    wait_for_version(service, 1)

    assert service.last_stats.steps == 32


def test_an_actor_cannot_submit_twice_in_one_round(start_learner, rollout):
    service, connect = start_learner(num_actors=2)
    client = connect("actor-a")
    client.get_weights()

    client.put(rollout)
    ack = client.put(rollout)

    assert not ack.accepted
    assert "already submitted" in ack.reason


def test_a_round_that_is_already_full_refuses_another_actor(start_learner, rollout):
    _, connect = start_learner(num_actors=1)  # no training thread, so the round stays full
    first, second = connect("actor-a"), connect("actor-b")
    first.get_weights()
    second.get_weights()

    first.put(rollout)
    ack = second.put(rollout)

    assert not ack.accepted
    assert "full" in ack.reason


def test_a_rollout_from_before_the_update_is_rejected_as_stale(start_learner, rollout):
    service, connect = start_learner()
    service.start_training()
    client = connect("actor-a")
    client.get_weights()
    client.put(rollout)
    wait_for_version(service, 1)

    ack = client.put(rollout)  # still tagged v0

    assert not ack.accepted
    assert "made with v0, learner is on v1" in ack.reason


def play(model, size=128):
    """A rollout played by model the way run_environment plays one."""
    buffer = make_buffer(model.observation_space, size=size)
    generator = th.Generator().manual_seed(2)
    for step in range(size):
        obs = {key: th.rand(1, *space.shape, generator=generator)
               for key, space in model.observation_space.spaces.items()}
        mask = mask_for(buffer, (step * 11) % 141)
        action, value, log_prob = act(model, obs, mask)
        buffer.add(obs, th.tensor(action), float(step % 3), step % 20 == 19, value, log_prob, mask)
    buffer.compute_returns_and_advantage(last_values=th.tensor([[0.0]]), dones=False)
    return buffer


def test_the_first_ratio_is_1_when_the_actor_played_with_the_learners_weights(start_learner):
    service, connect = start_learner()
    service.start_training()
    client = connect("actor-a")
    _, config, state_dict = client.get_weights()
    actor_model = make_model(config)
    actor_model.policy.load_state_dict(state_dict)

    client.put(play(actor_model))
    wait_for_version(service, 1)

    assert service.last_stats.first_ratio_error_max < 1e-4


def test_the_first_ratio_is_not_1_when_the_actor_played_with_other_weights(start_learner):
    service, connect = start_learner()
    service.start_training()
    client = connect("actor-a")
    _, config, _ = client.get_weights()

    client.put(play(make_model(config)))  # fresh random weights, the bug this phase fixes
    wait_for_version(service, 1)

    assert service.last_stats.first_ratio_error_max > 1e-2


def test_a_training_failure_takes_the_learner_down(start_learner, rollout, monkeypatch):
    exits = []
    monkeypatch.setattr(learner_server.os, "_exit", exits.append)
    monkeypatch.setattr(learner_server, "train", lambda *args: 1 / 0)
    service, connect = start_learner()
    service.start_training()
    client = connect("actor-a")
    client.get_weights()

    client.put(rollout)

    deadline = time.monotonic() + 10
    while not exits and time.monotonic() < deadline:
        time.sleep(0.01)
    assert exits == [1]
