"""Message framing between the actor and the middleman over TCP."""

import json
import socket
import threading

import pytest

import middleman_process
from util.communication import HEADER, MAX_MESSAGE_BYTES, FramedConnection


class FakeSocket:
    """Hands out scripted chunks from recv and records everything sent."""

    def __init__(self, chunks=()):
        self.chunks = list(chunks)
        self.sent = b""

    def recv(self, _bufsize):
        if not self.chunks:
            return b""
        chunk = self.chunks.pop(0)
        if isinstance(chunk, BaseException):
            raise chunk
        return chunk

    def sendall(self, data):
        self.sent += data


def wire_bytes(*messages):
    """The bytes FramedConnection puts on the wire for these messages."""
    sink = FakeSocket()
    connection = FramedConnection(sink)
    for message in messages:
        connection.send(message)
    return sink.sent


def test_two_messages_coalesced_into_one_read_are_both_received():
    first = json.dumps({"available_commands": ["play"], "in_game": True})
    second = json.dumps({"available_commands": ["proceed"], "in_game": True})
    connection = FramedConnection(FakeSocket([wire_bytes(first, second)]))

    assert connection.receive_json() == json.loads(first)
    assert connection.receive_json() == json.loads(second)


def test_one_message_split_across_several_reads_is_received_whole():
    message = json.dumps({"game_state": {"screen_type": "MAP", "floor": 3}})
    data = wire_bytes(message)
    chunks = [data[:2], data[2:7], data[7:20], data[20:]]
    connection = FramedConnection(FakeSocket(chunks))

    assert connection.receive_json() == json.loads(message)


def test_a_message_containing_json_delimiters_and_newlines_round_trips():
    message = 'PLAY 1 0\n{"not": "a boundary"}}'
    connection = FramedConnection(FakeSocket([wire_bytes(message)]))

    assert connection.receive() == message


def test_non_ascii_text_is_framed_by_byte_length():
    message = "Défense — 防御"
    connection = FramedConnection(FakeSocket([wire_bytes(message, "END")]))

    assert connection.receive() == message
    assert connection.receive() == "END"


def test_a_timeout_raises_instead_of_looping():
    data = wire_bytes("PROCEED")
    connection = FramedConnection(FakeSocket([data[:3], socket.timeout("timed out")]))

    with pytest.raises(TimeoutError):
        connection.receive()


def test_an_oversized_length_header_raises_instead_of_buffering():
    header = HEADER.pack(MAX_MESSAGE_BYTES + 1)
    connection = FramedConnection(FakeSocket([header, b"x" * 100]))

    with pytest.raises(ConnectionError):
        connection.receive()


def test_a_connection_closed_mid_message_raises():
    data = wire_bytes("PROCEED")
    connection = FramedConnection(FakeSocket([data[:5]]))

    with pytest.raises(ConnectionError):
        connection.receive()


class ScriptedStdin:
    """Stands in for the game's stdout: yields lines, then ends the relay."""

    def __init__(self, lines):
        self.lines = list(lines)

    def readline(self):
        if not self.lines:
            raise EOFError("script exhausted")
        return self.lines.pop(0)


def test_middleman_relays_states_and_commands_over_a_real_socket(monkeypatch, capsys):
    states = [{"available_commands": ["play"]}, {"available_commands": ["proceed"]}]
    monkeypatch.setattr(middleman_process.sys, "stdin", ScriptedStdin(json.dumps(s) + "\n" for s in states))
    monkeypatch.setattr(middleman_process, "log_message", lambda message: None)

    middleman_end, actor_end = socket.socketpair()
    actor = FramedConnection(actor_end)
    actor_end.settimeout(10)

    relay = threading.Thread(target=middleman_process.handle_gym_client, args=(middleman_end,))
    relay.start()
    try:
        assert actor.receive_json() == states[0]
        actor.send("PLAY 1 0")
        assert actor.receive_json() == states[1]
        actor.send("PROCEED")
        relay.join(timeout=10)
    finally:
        actor_end.close()

    assert not relay.is_alive()
    assert capsys.readouterr().out.splitlines() == ["PLAY 1 0", "PROCEED"]
