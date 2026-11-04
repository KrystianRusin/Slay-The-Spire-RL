import itertools
import os
import select
import socket
import sys
import json
import time

from util.communication import FramedConnection

# Set this to a directory to capture every game state that passes through the
# middleman. The captures are the raw material for tests/fixtures/game_states -
# play until you have hit the screens you need, then copy the interesting ones
# in. Unset (the default), capturing is off and training writes nothing extra.
CAPTURE_DIR_VAR = "STS_CAPTURE_DIR"

# Fixed port to listen on. Unset, the first free port from 9999 up is used, which
# is only safe when one game runs per host (e.g. one per container).
PORT_VAR = "STS_MIDDLEMAN_PORT"

# Set to 1 to log every game state and command. Off by default: at several
# steps per second the full states add up to gigabytes.
VERBOSE_VAR = "MIDDLEMAN_VERBOSE"
VERBOSE = os.environ.get(VERBOSE_VAR) == "1"

# Seconds without a message from the game before asking it to resend its state.
STALL_TIMEOUT = 10.0

_capture_sequence = itertools.count()

def log_message(message, verbose=False):
    """Log messages to a file. Per-step messages pass verbose=True and are skipped unless VERBOSE."""
    if verbose and not VERBOSE:
        return
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

def find_free_port(start_port=9999):
    """Finds a free port starting from `start_port` and increments by 1 until a free port is found."""
    port = start_port
    while True:
        try:
            # Try to bind to the given port
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("0.0.0.0", port))
                return port  # If successful, return the free port
        except OSError:
            log_message(f"Port {port} is in use, trying next port...")
            port += 1  # Increment the port number and try again

def read_lines_with_timeout(fd, timeout):
    """Yield lines from fd, or None whenever `timeout` seconds pass with no input.

    Reads the raw fd instead of sys.stdin, because select() can't see lines
    already sitting in Python's read buffer. Returns when the game closes the pipe.
    """
    buf = b""
    while True:
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            yield line.decode("utf-8", errors="replace")
        ready, _, _ = select.select([fd], [], [], timeout)
        if not ready:
            yield None
            continue
        chunk = os.read(fd, 65536)
        if not chunk:
            return
        buf += chunk

def send_to_game(command):
    sys.stdout.write(command + "\n")
    sys.stdout.flush()

def handle_gym_client(gym_client_socket, game_lines):
    """Relay game states to the gym client and its chosen commands back to the game.

    Two things never reach the client, so every message it gets is a real state
    to act on:
    - Stalls. The game sometimes drops a command sent while a screen is still
      animating in, then waits for us while we wait for it. After STALL_TIMEOUT
      of silence we send `state`, and forward the reply.
    - Errors. If the game rejects a command, we ask for the state again instead
      of handing the client an error message to act on.

    Returns True when the game has closed its output (time to shut down), or
    False when the client went away (wait for another one).
    """
    connection = FramedConnection(gym_client_socket)

    for game_state_json in game_lines:
        try:
            if game_state_json is None:
                log_message(f"No message from the game for {STALL_TIMEOUT}s; asking for its state.")
                send_to_game("state")
                continue
            game_state_json = game_state_json.strip()
            if not game_state_json:
                continue
            log_message(f"Received game state: {game_state_json}", verbose=True)

            # Parse the JSON game state
            try:
                game_state = json.loads(game_state_json)
            except json.JSONDecodeError:
                log_message("Received invalid JSON. Waiting for the next update...")
                continue

            if "error" in game_state:
                log_message(f"Game rejected the last command: {game_state['error']}; asking for its state.")
                send_to_game("state")
                continue

            save_game_state(game_state)

            connection.send(game_state_json)

            # The game sends nothing until it gets a command, so block without a timeout.
            command = connection.receive()
            log_message(f"Sent command to game: {command}", verbose=True)
            send_to_game(command)

        except Exception as e:
            log_message(f"Exception: {e}")
            gym_client_socket.close()
            return False

    log_message("The game closed its output; shutting down.")
    gym_client_socket.close()
    return True

def main():
    port = int(os.environ[PORT_VAR]) if os.environ.get(PORT_VAR) else find_free_port()

    # Create a TCP socket
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", port))
    server.listen(5)
    sys.stdout.write("ready\n")  # Communication Mod waits for this before sending game states
    sys.stdout.flush()
    log_message(f"Middleman process started and listening on port {port}.")
    # One reader for the whole process: a reconnecting client picks up where the last one left off.
    game_lines = read_lines_with_timeout(sys.stdin.fileno(), STALL_TIMEOUT)

    while True:
        try:
            # Accept a connection from the environment process
            client_socket, addr = server.accept()
            log_message(f"Accepted connection from {addr}")

            # Handle the gym client in the current thread to maintain continuous communication
            if handle_gym_client(client_socket, game_lines):
                break

        except Exception as e:
            log_message(f"Exception in main loop: {e}")
            break

if __name__ == "__main__":
    main()
