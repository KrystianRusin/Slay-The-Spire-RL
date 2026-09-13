"""The learner's checkpoint: policy weights and training progress, written as one archive.

Progress is an extra entry in the Stable Baselines3 archive, which
MaskablePPO.load ignores.
"""

import io
import json
import os
import zipfile
from collections import deque
from pathlib import Path

from sb3_contrib.ppo_mask import MaskablePPO

PROGRESS_ENTRY = "learner_progress.json"
# Far more than can be redelivered at once; see docs/adr/0003.
APPLIED_ROLLOUT_WINDOW = 10_000


class TrainingProgress:
    """Steps trained so far, and the IDs of the most recently applied rollouts."""

    def __init__(self, steps=0, applied_rollout_ids=()):
        self.steps = steps
        self._applied = deque(applied_rollout_ids, maxlen=APPLIED_ROLLOUT_WINDOW)

    def has_applied(self, rollout_id):
        return rollout_id in self._applied

    def record(self, rollout_id, steps):
        self._applied.append(rollout_id)
        self.steps += steps

    def to_json(self):
        return json.dumps({"steps": self.steps, "applied_rollout_ids": list(self._applied)})

    @classmethod
    def from_json(cls, text):
        data = json.loads(text)
        return cls(data["steps"], data["applied_rollout_ids"])


def save_checkpoint(model, progress, path):
    """Write the model and progress to path with a .zip suffix.

    The previous checkpoint is replaced only once the new one is fully written,
    so a process killed mid-save leaves the old one loadable.
    """
    target = _archive_path(path)
    archive = io.BytesIO()
    model.save(archive)
    with zipfile.ZipFile(archive, "a") as zipped:
        zipped.writestr(PROGRESS_ENTRY, progress.to_json())

    partial = target.with_name(target.name + ".partial")
    with open(partial, "wb") as file:
        file.write(archive.getvalue())
        file.flush()
        os.fsync(file.fileno())
    os.replace(partial, target)


def load_checkpoint(path, env, device):
    """Return (model, progress) from the checkpoint at path, or None if none has been saved.

    A model archive saved without progress restores with progress from zero.
    """
    target = _archive_path(path)
    if not target.exists():
        return None
    with zipfile.ZipFile(target) as zipped:
        if PROGRESS_ENTRY in zipped.namelist():
            progress = TrainingProgress.from_json(zipped.read(PROGRESS_ENTRY).decode("utf-8"))
        else:
            progress = TrainingProgress()
    return MaskablePPO.load(target, env=env, device=device), progress


def _archive_path(path):
    path = Path(path)
    return path if path.suffix == ".zip" else path.with_name(path.name + ".zip")
