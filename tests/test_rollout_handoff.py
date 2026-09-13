"""Handing a completed rollout from an actor to the learner."""

import numpy as np
import torch as th
from gymnasium import spaces

from environment.run_env import hand_off_rollout
from model.custom_rollout_buffer import CustomRolloutBuffer
from model.rollout_codec import decode_rollout

N_STEPS = 4
OBSERVATION_SPACE = spaces.Dict({"hand": spaces.Box(0, 1, shape=(3,), dtype=np.float32)})
ACTION_SPACE = spaces.Discrete(5)


class RecordingPublisher:
    def __init__(self):
        self.published = []

    def publish(self, encoded):
        self.published.append(encoded)


def _filled_buffer():
    buffer = CustomRolloutBuffer(N_STEPS, OBSERVATION_SPACE, ACTION_SPACE, device="cpu")
    for step in range(N_STEPS):
        buffer.add(
            {"hand": th.full((1, 3), step + 1.0)},
            th.tensor(step),
            float(step + 1),
            False,
            th.tensor(step + 0.5),
            th.tensor(-0.1),
        )
    return buffer


def test_published_rollout_survives_the_actor_starting_its_next_one():
    buffer = _filled_buffer()
    expected = {name: getattr(buffer, name).copy() for name in ["actions", "rewards", "values", "old_log_prob", "returns"]}
    expected_hand = buffer.observations["hand"].copy()
    publisher = RecordingPublisher()

    hand_off_rollout(buffer, publisher)
    buffer.add({"hand": th.zeros(1, 3)}, th.tensor(0), 0.0, False, th.tensor(0.0), th.tensor(0.0))

    (encoded,) = publisher.published
    received = decode_rollout(encoded, OBSERVATION_SPACE, ACTION_SPACE)

    assert len(received) == N_STEPS
    np.testing.assert_array_equal(received.observations["hand"], expected_hand)
    for name, values in expected.items():
        np.testing.assert_array_equal(getattr(received, name), values, err_msg=name)
    assert len(buffer) == 1
