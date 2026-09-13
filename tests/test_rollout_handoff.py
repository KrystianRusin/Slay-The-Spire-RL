"""Handing a completed rollout from an actor to the learner's queue."""

import pickle
import threading
from multiprocessing import Queue

import numpy as np
import torch as th
from gymnasium import spaces

from environment.run_env import hand_off_rollout
from model.custom_rollout_buffer import CustomRolloutBuffer
from model.rollout_codec import decode_rollout

N_STEPS = 4
OBSERVATION_SPACE = spaces.Dict({"hand": spaces.Box(0, 1, shape=(3,), dtype=np.float32)})
ACTION_SPACE = spaces.Discrete(5)

_release_feeder = threading.Event()


class _BlocksFeeder:
    """Stalls the queue's feeder thread while it pickles this object."""

    def __reduce__(self):
        assert _release_feeder.wait(timeout=10), "feeder was never released"
        return (_BlocksFeeder, ())


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


def test_queued_rollout_survives_the_actor_starting_its_next_one():
    buffer = _filled_buffer()
    expected = pickle.loads(pickle.dumps(buffer))
    queue = Queue()

    _release_feeder.clear()
    queue.put(_BlocksFeeder())
    hand_off_rollout(buffer, queue)
    _release_feeder.set()

    queue.get(timeout=10)
    received = decode_rollout(queue.get(timeout=10), OBSERVATION_SPACE, ACTION_SPACE)

    assert len(received) == N_STEPS
    np.testing.assert_array_equal(received.observations["hand"], expected.observations["hand"])
    np.testing.assert_array_equal(received.actions, expected.actions)
    np.testing.assert_array_equal(received.rewards, expected.rewards)
    np.testing.assert_array_equal(received.values, expected.values)
    np.testing.assert_array_equal(received.old_log_prob, expected.old_log_prob)
    np.testing.assert_array_equal(received.returns, expected.returns)
    assert len(buffer) == 0
