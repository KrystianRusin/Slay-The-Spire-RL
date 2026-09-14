"""The middleman: binding its port once, registering its game, and handing the game over to a reconnecting actor."""

import io
import json
import os
import queue
import socket
import subprocess
import sys
import threading

import pytest

import middleman_process
from db.game_registry import GameRegistry, wait_for_game
from util.communication import Backoff, FramedConnection, GameConnection

from tests.conftest import REPO_ROOT
from tests.test_data_layer import database  # noqa: F401
from tests.test_framing import ScriptedGame, relay

GAME_ID = "game-under-test"
TIMEOUT_SECONDS = 30


def test_listening_binds_once_and_reports_the_port_it_got():
    server, port = middleman_process.listen(0)
    try:
        assert port > 0
        socket.create_connection(("127.0.0.1", port), timeout=5).close()
    finally:
        server.close()


def test_a_configured_port_already_in_use_fails_instead_of_moving_to_another():
    taken, port = middleman_process.listen(0)
    try:
        with pytest.raises(OSError):
            middleman_process.listen(port)
    finally:
        taken.close()


def test_the_game_id_defaults_to_this_host_and_the_game_process_that_started_the_middleman(monkeypatch):
    monkeypatch.delenv(middleman_process.GAME_ID_VAR, raising=False)
    assert middleman_process.own_game_id() == f"{socket.gethostname()}-{os.getppid()}"

    monkeypatch.setenv(middleman_process.GAME_ID_VAR, "game-7")
    assert middleman_process.own_game_id() == "game-7"


def test_a_state_the_dropped_client_never_answered_is_sent_to_the_next_one(monkeypatch, capsys):
    state = json.dumps({"available_commands": ["play"]})
    game = ScriptedGame([state + "\n"])
    monkeypatch.setattr(middleman_process, "log_message", lambda message: None)

    middleman_end, actor_end = socket.socketpair()
    actor_end.settimeout(10)
    first = relay(middleman_end, game)
    assert FramedConnection(actor_end).receive() == state
    actor_end.close()
    assert first.get(timeout=10) == state

    middleman_end, actor_end = socket.socketpair()
    actor_end.settimeout(10)
    actor = FramedConnection(actor_end)
    second = relay(middleman_end, game, pending_state=state)
    try:
        assert actor.receive() == state
        actor.send("PLAY 1 0")
        assert isinstance(second.get(timeout=10), middleman_process.GameClosed)
    finally:
        actor_end.close()

    assert capsys.readouterr().out.splitlines() == ["PLAY 1 0"]


class ClosedGameOutput:
    def write(self, text):
        raise BrokenPipeError("the game is gone")

    def flush(self):
        pass


def test_a_command_the_game_cannot_take_stops_the_relay_instead_of_waiting_for_another_client(monkeypatch):
    monkeypatch.setattr(middleman_process.sys, "stdout", ClosedGameOutput())
    monkeypatch.setattr(middleman_process, "log_message", lambda message: None)

    middleman_end, actor_end = socket.socketpair()
    actor_end.settimeout(10)
    actor = FramedConnection(actor_end)
    outcome = relay(middleman_end, ScriptedGame(['{"available_commands": ["play"]}\n']))
    try:
        actor.receive()
        actor.send("PLAY 1 0")
        assert isinstance(outcome.get(timeout=10), BrokenPipeError)
    finally:
        actor_end.close()


def test_game_input_is_read_ahead_and_reports_when_the_game_closes():
    game_input = middleman_process.GameInput(io.StringIO("first\nsecond\n"))

    assert game_input.closed.wait(timeout=10)
    assert [game_input.readline(), game_input.readline(), game_input.readline(), game_input.readline()] == ["first\n", "second\n", "", ""]


class FakeGame:
    """Plays the game's side of a real middleman process: its stdin and stdout."""

    def __init__(self, cwd):
        env = {
            **os.environ,
            middleman_process.GAME_ID_VAR: GAME_ID,
            middleman_process.ADVERTISED_HOST_VAR: "127.0.0.1",
        }
        self.process = subprocess.Popen(
            [sys.executable, str(REPO_ROOT / "middleman_process.py")],
            cwd=cwd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
        )
        self._lines = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.process.stdout:
            self._lines.put(line.rstrip("\n"))

    def read_line(self):
        return self._lines.get(timeout=TIMEOUT_SECONDS)

    def send_state(self, state):
        self.process.stdin.write(json.dumps(state) + "\n")
        self.process.stdin.flush()

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                self.process.kill()
                raise

    def kill(self):
        self.process.kill()
        self.process.wait(timeout=TIMEOUT_SECONDS)


def test_killing_and_restarting_the_middleman_mid_episode_resumes_the_episode(database, tmp_path):
    in_combat = {"available_commands": ["play", "end"], "game_state": {"floor": 1}}
    after_play = {"available_commands": ["play", "end"], "game_state": {"floor": 1, "turn": 2}}
    registry = GameRegistry()

    game = FakeGame(tmp_path)
    try:
        assert game.read_line() == "ready"
        game_id = wait_for_game(registry, "actor-0", poll_seconds=0.05)
        connection = GameConnection(
            lambda: registry.address(game_id, "actor-0"),
            Backoff(initial_seconds=0.05, max_seconds=0.5, max_retries=60),
        )

        game.send_state(in_combat)
        assert connection.receive_json() == in_combat
        connection.send("PLAY 1 0")
        assert game.read_line() == "PLAY 1 0"
        game.send_state(after_play)
        assert connection.receive_json() == after_play

        game.kill()
        game = FakeGame(tmp_path)
        assert game.read_line() == "ready"
        # Communication Mod sends the state it is waiting on again once a new process is ready.
        game.send_state(after_play)

        assert connection.receive_json() == after_play
        connection.send("END")
        assert game.read_line() == "END"
        connection.close()
    finally:
        game.close()

    assert game.process.returncode == 0
    assert registry.address(game_id, "actor-0") is None


def test_a_game_that_closes_while_no_actor_is_connected_is_removed_from_the_registry(database, tmp_path):
    registry = GameRegistry()
    game = FakeGame(tmp_path)
    try:
        assert game.read_line() == "ready"
        wait_for_game(registry, "actor-0", poll_seconds=0.05)
        registry.release(GAME_ID, "actor-0")
    finally:
        game.close()

    assert game.process.returncode == 0
    assert registry.claim("actor-0") is None
