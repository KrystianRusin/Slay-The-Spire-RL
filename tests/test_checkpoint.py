"""The learner saves its state after every update and resumes from it on restart."""
import os

import numpy as np
import pytest
import torch as th

pytest.importorskip("rpc.learner_pb2", reason="generate it from rpc/learner.proto first")

from learner_server import LearnerService  # noqa: E402
from model import checkpoint  # noqa: E402
from model.policy import TrainingConfig, make_model  # noqa: E402
from rpc.codec import rollout_from_proto, rollout_to_proto, state_dict_digest  # noqa: E402
from tests.rollouts import play  # noqa: E402

CONFIG = TrainingConfig(n_steps=128, policy_layers=(32,), value_layers=(16,))


def played_by(service):
    return rollout_from_proto(rollout_to_proto(play(service.model), "actor-a", service.version))


def train_one_round(service, seed, rollout=None):
    """One update, by default on a rollout played with the service's current weights."""
    rollout = rollout or played_by(service)
    np.random.seed(seed)  # the minibatch order
    service._last_arrival = service._first_arrival = 0.0
    service._train_round([rollout])


def optimizer_tensors(service):
    state = service.policy.optimizer.state_dict()["state"]
    return [tensor for slot in state.values() for tensor in slot.values() if th.is_tensor(tensor)]


def test_without_a_checkpoint_folder_it_starts_fresh_and_saves_nothing(tmp_path):
    service = LearnerService(CONFIG)
    train_one_round(service, seed=0)

    assert service.version == 1
    assert list(tmp_path.iterdir()) == []


def test_a_restarted_learner_resumes_with_the_same_weights_and_optimizer(tmp_path):
    first = LearnerService(CONFIG, checkpoint_dir=str(tmp_path))
    train_one_round(first, seed=0)

    resumed = LearnerService(CONFIG, checkpoint_dir=str(tmp_path))

    assert resumed.version == 1
    assert state_dict_digest(resumed.policy.state_dict()) == state_dict_digest(first.policy.state_dict())
    assert resumed._weights == first._weights  # what actors get served
    saved, loaded = optimizer_tensors(first), optimizer_tensors(resumed)
    assert saved and len(saved) == len(loaded)
    assert all(th.equal(a, b) for a, b in zip(saved, loaded))


def test_resuming_trains_exactly_like_never_stopping(tmp_path):
    kept_running = LearnerService(CONFIG, checkpoint_dir=str(tmp_path / "a"))
    train_one_round(kept_running, seed=0)
    resumed = LearnerService(CONFIG, checkpoint_dir=str(tmp_path / "b"))
    checkpoint.restore(resumed.model, checkpoint.load_latest(str(tmp_path / "a")))
    resumed.version = 1
    # Same weights but a fresh optimizer: what we'd get without saving its state.
    weights_only = LearnerService(CONFIG)
    weights_only.policy.load_state_dict(kept_running.policy.state_dict())
    weights_only.version = 1

    rollout = played_by(kept_running)  # the same steps for all three
    for service in (kept_running, resumed, weights_only):
        train_one_round(service, seed=1, rollout=rollout)

    digest = state_dict_digest(kept_running.policy.state_dict())
    assert state_dict_digest(resumed.policy.state_dict()) == digest
    assert state_dict_digest(weights_only.policy.state_dict()) != digest


def test_only_the_newest_three_are_kept(tmp_path):
    model = make_model(CONFIG)
    for version in range(1, 6):
        checkpoint.save_checkpoint(str(tmp_path), version, model, CONFIG)

    assert [v for v, _ in checkpoint.list_checkpoints(str(tmp_path))] == [3, 4, 5]
    assert sorted(os.listdir(tmp_path)) == ["v000003.pt", "v000004.pt", "v000005.pt"]


def test_a_half_written_save_is_ignored_then_cleaned_up(tmp_path):
    model = make_model(CONFIG)
    checkpoint.save_checkpoint(str(tmp_path), 1, model, CONFIG)
    (tmp_path / "v000002.pt.tmp").write_bytes(b"the learner died here")

    assert checkpoint.load_latest(str(tmp_path))["version"] == 1

    checkpoint.save_checkpoint(str(tmp_path), 2, model, CONFIG)
    assert sorted(os.listdir(tmp_path)) == ["v000001.pt", "v000002.pt"]


def test_a_different_config_is_refused_and_named(tmp_path):
    train_one_round(LearnerService(CONFIG, checkpoint_dir=str(tmp_path)), seed=0)
    changed = TrainingConfig(n_steps=256, gamma=0.9, policy_layers=(32,), value_layers=(16,))

    with pytest.raises(checkpoint.ConfigMismatch, match="n_steps.*gamma"):
        LearnerService(changed, checkpoint_dir=str(tmp_path))


def test_the_checkpoint_is_on_disk_before_actors_can_get_the_version(tmp_path, monkeypatch):
    service = LearnerService(CONFIG, checkpoint_dir=str(tmp_path))
    published_when_saved = []
    save = checkpoint.save_checkpoint

    def recording_save(directory, version, *args, **kwargs):
        published_when_saved.append(service.version)
        return save(directory, version, *args, **kwargs)

    monkeypatch.setattr(checkpoint, "save_checkpoint", recording_save)
    train_one_round(service, seed=0)

    assert published_when_saved == [0]  # v1 was saved while actors were still served v0
    assert service.version == 1
