"""The learner's checkpoint: policy weights and training progress, saved and restored together."""

import pytest
import torch as th

from model.checkpoint import TrainingProgress, load_checkpoint, save_checkpoint
from slay_the_spire_env import SlayTheSpireEnv

from tests.test_learner import make_model, weights


def test_weights_and_progress_are_restored_together(tmp_path):
    model = make_model()
    progress = TrainingProgress()
    progress.record("rollout-a", steps=2048)
    progress.record("rollout-b", steps=1024)

    save_checkpoint(model, progress, tmp_path / "policy")
    restored_model, restored = load_checkpoint(tmp_path / "policy", SlayTheSpireEnv({}), device="cpu")

    assert restored.steps == 3072
    assert restored.has_applied("rollout-a") and restored.has_applied("rollout-b")
    assert not restored.has_applied("rollout-c")
    assert all(th.equal(old, new) for old, new in zip(weights(model), restored_model.policy.parameters()))


def test_a_model_saved_without_progress_restores_its_weights_with_progress_from_zero(tmp_path):
    model = make_model()
    model.save(tmp_path / "policy")

    restored_model, restored = load_checkpoint(tmp_path / "policy", SlayTheSpireEnv({}), device="cpu")

    assert restored.steps == 0
    assert all(th.equal(old, new) for old, new in zip(weights(model), restored_model.policy.parameters()))


def test_there_is_nothing_to_restore_before_the_first_save(tmp_path):
    assert load_checkpoint(tmp_path / "policy", SlayTheSpireEnv({}), device="cpu") is None


def test_a_save_interrupted_partway_leaves_the_previous_checkpoint(tmp_path, monkeypatch):
    model = make_model()
    progress = TrainingProgress()
    progress.record("rollout-a", steps=2048)
    save_checkpoint(model, progress, tmp_path / "policy")

    def dies_while_writing(path_or_file, *args, **kwargs):
        path_or_file.write(b"PK\x03\x04 half an archive")
        raise KeyboardInterrupt

    progress.record("rollout-b", steps=2048)
    monkeypatch.setattr(model, "save", dies_while_writing)
    with pytest.raises(KeyboardInterrupt):
        save_checkpoint(model, progress, tmp_path / "policy")

    _, restored = load_checkpoint(tmp_path / "policy", SlayTheSpireEnv({}), device="cpu")
    assert restored.steps == 2048
    assert not restored.has_applied("rollout-b")


def test_the_policy_version_counts_updates_and_survives_a_restart(tmp_path):
    progress = TrainingProgress()
    progress.record("rollout-a", steps=2048)
    progress.record("rollout-b", steps=2048)

    save_checkpoint(make_model(), progress, tmp_path / "policy")
    _, restored = load_checkpoint(tmp_path / "policy", SlayTheSpireEnv({}), device="cpu")

    assert restored.policy_version == 2
