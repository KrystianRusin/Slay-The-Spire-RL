"""Save and load the learner's training state, so a restart carries on where it stopped.

A checkpoint holds the policy weights, the optimizer state (Adam's running
averages), the version number and the config. Files are named v000123.pt and
only the newest few are kept.

Each file is written under a temporary name and then renamed. A rename is all
or nothing, so a crash part way through a save leaves the previous checkpoints
intact instead of a half-written file that the next start would try to load.
"""
import dataclasses
import os
import re

import torch

from model.policy import TrainingConfig

KEEP = 3
_NAME = re.compile(r"^v(\d+)\.pt$")
_TEMP_SUFFIX = ".tmp"


class ConfigMismatch(Exception):
    """The checkpoint was trained with a different config than the one asked for."""


def checkpoint_path(directory, version):
    return os.path.join(directory, f"v{version:06d}.pt")


def list_checkpoints(directory):
    """(version, path) of every finished checkpoint in directory, oldest first."""
    if not os.path.isdir(directory):
        return []
    found = []
    for name in os.listdir(directory):
        match = _NAME.match(name)
        if match:
            found.append((int(match.group(1)), os.path.join(directory, name)))
    return sorted(found)


def save_checkpoint(directory, version, model, config, keep=KEEP):
    """Write version's checkpoint, then delete all but the newest keep. Returns its path."""
    os.makedirs(directory, exist_ok=True)
    path = checkpoint_path(directory, version)
    temp = path + _TEMP_SUFFIX
    state = {
        "version": version,
        "config": dataclasses.asdict(config),
        "policy": model.policy.state_dict(),
        "optimizer": model.policy.optimizer.state_dict(),
    }
    with open(temp, "wb") as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())  # on disk before the rename makes it the newest
    os.replace(temp, path)

    for _, old in list_checkpoints(directory)[:-keep]:
        os.remove(old)
    for name in os.listdir(directory):  # leftovers from a save that crashed
        if name.endswith(_TEMP_SUFFIX):
            os.remove(os.path.join(directory, name))
    return path


def load_latest(directory):
    """The newest checkpoint as a dict, or None if there isn't one."""
    checkpoints = list_checkpoints(directory)
    if not checkpoints:
        return None
    _, path = checkpoints[-1]
    # weights_only: only tensors and plain containers, never arbitrary pickled objects.
    state = torch.load(path, weights_only=True)
    state["config"] = TrainingConfig(**{
        key: tuple(value) if isinstance(value, list) else value
        for key, value in state["config"].items()
    })
    state["path"] = path
    return state


def check_config(saved, wanted):
    """Raise ConfigMismatch naming every field where saved and wanted differ."""
    saved_fields, wanted_fields = dataclasses.asdict(saved), dataclasses.asdict(wanted)
    differences = [f"{key}: checkpoint has {saved_fields[key]}, asked for {wanted_fields[key]}"
                   for key in wanted_fields if saved_fields.get(key) != wanted_fields[key]]
    if differences:
        raise ConfigMismatch("; ".join(differences))


def restore(model, state):
    """Load a checkpoint's weights and optimizer state into model."""
    model.policy.load_state_dict(state["policy"])
    model.policy.optimizer.load_state_dict(state["optimizer"])
