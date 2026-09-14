"""The middleman: started by Communication Mod for one game instance, it relays game states to the actor that claimed the game and commands back.

It binds its port once, registers the game and that port in the game
registry, and keeps the registration alive while it runs. Nothing but
commands for the game and the ready handshake may be written to stdout.
"""

import itertools
import os
import queue
import socket
import sys
import json
import threading
import time

from dotenv import load_dotenv

from db.game_registry import HEARTBEAT_SECONDS, GameRegistry
from db.session import init_db
from util.communication import FramedConnection
from util.heartbeat import Heartbeat

# Set this to a directory to capture every game state that passes through the
# middleman. The captures are the raw material for tests/fixtures/game_states -
# play until you have hit the screens you need, then copy the interesting ones
# in. Unset (the default), capturing is off and training writes nothing extra.
CAPTURE_DIR_VAR = "STS_CAPTURE_DIR"
# Identifies this game instance in the registry. Only needed when the game does not start the middleman directly.
GAME_ID_VAR = "STS_GAME_ID"
# The port to listen on. Unset, the middleman takes any free port and registers whichever it got.
PORT_VAR = "STS_MIDDLEMAN_PORT"
# The host actors should dial to reach this middleman, registered alongside its port.
ADVERTISED_HOST_VAR = "STS_ADVERTISED_HOST"
ACCEPT_POLL_SECONDS = 1.0

_capture_sequence = itertools.count()

def log_message(message):
    """Log messages to a file."""
    with open("middleman_log.txt", "a") as log_file:
        log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

def save_game_state(state):
    """Write one game state to the capture directory, if capturing is enabled.

    Named by screen type so the screens you are missing are obvious from a
    directory listing, and pretty-printed so a committed fixture is reviewable.
    """
    capture_dir = os.environ.get(CAPTURE_DIR_VAR)
    if not capture_dir:
        return

    game_state = state.get("game_state")
    if isinstance(game_state, dict):
        screen = game_state.get("screen_type") or "unknown"
    else:
        screen = "no_game"

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    sequence = next(_capture_sequence)
    path = os.path.join(capture_dir, f"{screen.lower()}_{timestamp}_{sequence:04d}.json")

    try:
        os.makedirs(capture_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
            f.write("\n")
    except OSError as e:
        # Capturing is a debugging aid; never let it interrupt a run.
        log_message(f"Could not capture game state to {path}: {e}")

def own_game_id():
    """This game instance's id: STS_GAME_ID if set, otherwise this host and the game process that started the middleman.

    Communication Mod starts the middleman as a direct child of the game, so
    a middleman it restarts keeps the same id.
    """
    return os.environ.get(GAME_ID_VAR) or f"{socket.gethostname()}-{os.getppid()}"

def listen(port=0):
    """Bind and listen on port, 0 for any free one, returning the socket and the port actually bound."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        server.bind(("0.0.0.0", port))
        server.listen(5)
    except OSError:
        server.close()
        raise
    return server, server.getsockname()[1]

class GameClosed(Exception):
    """The game closed the middleman's input, so there is nothing left to relay."""

class GameInput:
    """Lines from the game, read ahead on a background thread so the game closing is noticed even while no actor is connected."""

    def __init__(self, stream):
        self.closed = threading.Event()
        self._lines = queue.Queue()
        threading.Thread(target=self._read, args=(stream,), name="game-input", daemon=True).start()

    def readline(self):
        """The next line from the game, or an empty string once it has closed."""
        if self.closed.is_set() and self._lines.empty():
            return ""
        return self._lines.get()

    def _read(self, stream):
        for line in iter(stream.readline, ""):
            self._lines.put(line)
        self.closed.set()
        self._lines.put("")

def read_game_state(game_input):
    """Block until the game sends a valid game state, and return its line of JSON. Raises GameClosed at end of input."""
    while True:
        line = game_input.readline()
        if not line:
            raise GameClosed("The game closed its output")
        game_state_json = line.strip()
        log_message(f"Received game state: {game_state_json}")
        if not game_state_json:
            continue
        try:
            game_state = json.loads(game_state_json)
        except json.JSONDecodeError:
            log_message("Received invalid JSON. Waiting for the next update...")
            continue
        save_game_state(game_state)
        return game_state_json

def handle_gym_client(gym_client_socket, game_input, pending_state=None):
    """Relay game states to one gym client and its chosen commands back to the game, until the client drops.

    The game sends each state once and then waits for a command, so a state
    sent to a client that dropped before answering is passed in as
    pending_state and sent to the next client first. Returns the state left
    unanswered when this client drops, or None. Raises GameClosed when the
    game's input ends, and whatever writing to the game raises.
    """
    connection = FramedConnection(gym_client_socket)
    try:
        while True:
            if pending_state is None:
                pending_state = read_game_state(game_input)
            try:
                connection.send(pending_state)
                # The game sends nothing until it gets a command, so block without a timeout.
                command = connection.receive()
            except OSError as e:
                log_message(f"Lost the gym client: {e}")
                return pending_state
            log_message(f"Received command from gym client: {command}")

            sys.stdout.write(command + "\n")
            sys.stdout.flush()
            log_message(f"Sent command to game: {command}")
            pending_state = None
    finally:
        gym_client_socket.close()

def accept_while_game_open(server, game_input):
    """Block until a client connects, and return its socket. Raises GameClosed if the game closes first."""
    server.settimeout(ACCEPT_POLL_SECONDS)
    while not game_input.closed.is_set():
        try:
            client_socket, addr = server.accept()
        except TimeoutError:
            continue
        client_socket.settimeout(None)
        log_message(f"Accepted connection from {addr}")
        return client_socket
    raise GameClosed("The game closed while no client was connected")

def main():
    load_dotenv()
    init_db()
    server, port = listen(int(os.environ.get(PORT_VAR, "0")))
    registry = GameRegistry()
    game_id = own_game_id()
    host = os.environ.get(ADVERTISED_HOST_VAR, "localhost")
    registry.register(game_id, host, port)
    log_message(f"Middleman for game {game_id} started and listening on port {port}, registered as {host}:{port}.")

    game_input = GameInput(sys.stdin)
    try:
        with Heartbeat(HEARTBEAT_SECONDS, lambda: registry.register(game_id, host, port), "game registration"):
            sys.stdout.write("ready\n")  # Communication Mod waits for this before sending game states
            sys.stdout.flush()
            pending_state = None
            while True:
                client_socket = accept_while_game_open(server, game_input)
                pending_state = handle_gym_client(client_socket, game_input, pending_state)
    except GameClosed:
        log_message("The game closed; stopping.")
    finally:
        server.close()
        registry.deregister(game_id, port)

if __name__ == "__main__":
    main()
