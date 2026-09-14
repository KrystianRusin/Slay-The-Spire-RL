"""A function called periodically on a background thread."""

import threading

from util.heartbeat import Heartbeat


def test_beats_repeat_until_the_block_ends_and_survive_a_failing_beat():
    beats = []
    three_beats = threading.Event()

    def beat():
        beats.append(len(beats))
        if len(beats) == 3:
            three_beats.set()
        if len(beats) == 1:
            raise RuntimeError("database unavailable")

    with Heartbeat(0.01, beat, "test"):
        assert three_beats.wait(timeout=10)
    stopped_at = len(beats)
    threading.Event().wait(0.05)

    assert len(beats) == stopped_at
