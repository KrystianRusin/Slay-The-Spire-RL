"""The actor's connection to its game, re-established with backoff when it drops."""

import pytest

from util.communication import Backoff, GameConnection

from tests.test_framing import FakeSocket, wire_bytes

ADDRESS = ("localhost", 9999)


class ClosableSocket(FakeSocket):
    def __init__(self, chunks=(), send_error=None):
        super().__init__(chunks)
        self.closed = False
        self.send_error = send_error

    def sendall(self, data):
        if self.send_error:
            raise self.send_error
        super().sendall(data)

    def close(self):
        self.closed = True


def game_connection(*outcomes, max_seconds=30.0):
    """A GameConnection whose connection attempts produce outcomes in turn, recording its sleeps."""
    remaining = list(outcomes)
    sleeps = []

    def connect(address):
        assert address == ADDRESS
        outcome = remaining.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    backoff = Backoff(initial_seconds=1.0, max_seconds=max_seconds, sleep=sleeps.append)
    return GameConnection(ADDRESS, backoff, connect=connect), sleeps


def test_a_dropped_connection_is_reopened_after_growing_delays_instead_of_spinning():
    dropped = [ClosableSocket() for _ in range(4)]
    connection, sleeps = game_connection(*dropped, ClosableSocket([wire_bytes("state")]), max_seconds=3.0)

    assert connection.receive() == "state"
    assert sleeps == [1.0, 2.0, 3.0, 3.0]
    assert all(sock.closed for sock in dropped)


def test_the_delay_starts_over_once_a_message_arrives():
    connection, sleeps = game_connection(
        ClosableSocket(),
        ClosableSocket([wire_bytes("first")]),
        ClosableSocket([wire_bytes("second")]),
    )

    assert connection.receive() == "first"
    assert connection.receive() == "second"
    assert sleeps == [1.0, 1.0]


def test_a_refused_connection_is_retried_with_backoff():
    connection, sleeps = game_connection(
        ConnectionRefusedError(),
        ConnectionRefusedError(),
        ClosableSocket([wire_bytes("state")]),
    )

    assert connection.receive() == "state"
    assert sleeps == [1.0, 2.0]


def test_a_failed_send_raises_and_the_next_receive_reconnects():
    broken = ClosableSocket([wire_bytes("state")], send_error=ConnectionResetError())
    connection, sleeps = game_connection(broken, ClosableSocket([wire_bytes("next state")]))

    assert connection.receive() == "state"
    with pytest.raises(ConnectionError):
        connection.send("PROCEED")

    assert broken.closed
    assert connection.receive() == "next state"
    assert sleeps == [1.0]
