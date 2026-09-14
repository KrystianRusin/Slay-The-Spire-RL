"""The actor's connection to its game, re-established with backoff when it drops, and given up after bounded retries."""

import pytest

from db.game_registry import ClaimLost
from util.communication import Backoff, GameConnection, GameUnreachable

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


def game_connection(*outcomes, max_seconds=30.0, max_retries=None, locate=lambda: ADDRESS):
    """A GameConnection whose connection attempts produce outcomes in turn, recording its sleeps and the addresses it dials."""
    remaining = list(outcomes)
    sleeps = []
    dialled = []

    def connect(address):
        dialled.append(address)
        outcome = remaining.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    backoff = Backoff(initial_seconds=1.0, max_seconds=max_seconds, max_retries=max_retries, sleep=sleeps.append)
    return GameConnection(locate, backoff, connect=connect), sleeps, dialled


def test_a_dropped_connection_is_reopened_after_growing_delays_instead_of_spinning():
    dropped = [ClosableSocket() for _ in range(4)]
    connection, sleeps, _ = game_connection(*dropped, ClosableSocket([wire_bytes("state")]), max_seconds=3.0)

    assert connection.receive() == "state"
    assert sleeps == [1.0, 2.0, 3.0, 3.0]
    assert all(sock.closed for sock in dropped)


def test_the_delay_starts_over_once_a_message_arrives():
    connection, sleeps, _ = game_connection(
        ClosableSocket(),
        ClosableSocket([wire_bytes("first")]),
        ClosableSocket([wire_bytes("second")]),
    )

    assert connection.receive() == "first"
    assert connection.receive() == "second"
    assert sleeps == [1.0, 1.0]


def test_a_refused_connection_is_retried_with_backoff():
    connection, sleeps, _ = game_connection(
        ConnectionRefusedError(),
        ConnectionRefusedError(),
        ClosableSocket([wire_bytes("state")]),
    )

    assert connection.receive() == "state"
    assert sleeps == [1.0, 2.0]


def test_a_failed_send_raises_and_the_next_receive_reconnects():
    broken = ClosableSocket([wire_bytes("state")], send_error=ConnectionResetError())
    connection, sleeps, _ = game_connection(broken, ClosableSocket([wire_bytes("next state")]))

    assert connection.receive() == "state"
    with pytest.raises(ConnectionError):
        connection.send("PROCEED")

    assert broken.closed
    assert connection.receive() == "next state"
    assert sleeps == [1.0]


def test_the_game_is_located_again_for_every_reconnection():
    addresses = [("localhost", 5001), ("localhost", 6001)]
    connection, _, dialled = game_connection(
        ClosableSocket([wire_bytes("before restart")]),
        ClosableSocket([wire_bytes("after restart")]),
        locate=lambda: addresses[0],
    )

    assert connection.receive() == "before restart"
    addresses.pop(0)
    assert connection.receive() == "after restart"
    assert dialled == [("localhost", 5001), ("localhost", 6001)]


def test_waiting_for_the_game_to_be_located_counts_as_a_retry():
    locations = [None, None, ADDRESS]
    connection, sleeps, _ = game_connection(ClosableSocket([wire_bytes("state")]), locate=lambda: locations.pop(0))

    assert connection.receive() == "state"
    assert sleeps == [1.0, 2.0]


def test_an_unreachable_game_fails_loudly_after_the_retries_run_out():
    connection, sleeps, _ = game_connection(*[ConnectionRefusedError()] * 4, max_retries=3)

    with pytest.raises(GameUnreachable):
        connection.receive()
    assert sleeps == [1.0, 2.0, 4.0]


def test_the_retry_budget_starts_over_once_a_message_arrives():
    connection, _, _ = game_connection(
        ConnectionRefusedError(),
        ConnectionRefusedError(),
        ClosableSocket([wire_bytes("state")]),
        ConnectionRefusedError(),
        ConnectionRefusedError(),
        ClosableSocket([wire_bytes("next state")]),
        max_retries=3,
    )

    assert connection.receive() == "state"
    assert connection.receive() == "next state"


def test_an_unreachable_game_is_not_mistaken_for_a_dropped_send():
    connection, _, _ = game_connection(*[ConnectionRefusedError()] * 2, max_retries=1)

    with pytest.raises(GameUnreachable) as raised:
        connection.send("PROCEED")
    assert not isinstance(raised.value, ConnectionError)


def test_an_abandoned_connection_raises_on_the_next_send_or_receive():
    sock = ClosableSocket([wire_bytes("state", "next state")])
    connection, _, _ = game_connection(sock)
    assert connection.receive() == "state"

    connection.abandon(ClaimLost("taken"))

    with pytest.raises(ClaimLost):
        connection.receive()
    with pytest.raises(ClaimLost):
        connection.send("PROCEED")
    assert sock.closed


def test_a_claim_lost_to_another_actor_stops_the_connection():
    def locate():
        raise ClaimLost("taken")

    connection, sleeps, _ = game_connection(locate=locate)

    with pytest.raises(ClaimLost):
        connection.receive()
    assert sleeps == []
