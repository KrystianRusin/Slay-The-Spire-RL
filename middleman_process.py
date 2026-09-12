import itertools
import os
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

def handle_gym_client(gym_client_socket):
    """Relay game states to the gym client and its chosen commands back to the game."""
    connection = FramedConnection(gym_client_socket)

    while True:
        try:
            # Read the game state from stdin
            game_state_json = sys.stdin.readline().strip()
            log_message(f"Received game state: {game_state_json}")
            if not game_state_json:
                log_message("No game state received, waiting for the next update.")
                time.sleep(0.1)  # Reduced sleep time
                continue

            # Parse the JSON game state
            try:
                game_state = json.loads(game_state_json)
            except json.JSONDecodeError:
                log_message("Received invalid JSON. Waiting for the next update...")
                continue

            save_game_state(game_state)

            connection.send(game_state_json)

            # The game sends nothing until it gets a command, so block without a timeout.
            command = connection.receive()
            log_message(f"Received command from gym client: {command}")

            sys.stdout.write(command + "\n")
            sys.stdout.flush()
            log_message(f"Sent command to game: {command}")

        except Exception as e:
            log_message(f"Exception: {e}")
            break

    gym_client_socket.close()

def main():
    # Find a free port starting from 9999
    port = find_free_port()
    
    # Create a TCP socket
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("0.0.0.0", port))
    server.listen(5)
    sys.stdout.write("ready\n")  # Communication Mod waits for this before sending game states
    sys.stdout.flush()
    log_message(f"Middleman process started and listening on port {port}.")

    while True:
        try:
            # Accept a connection from the environment process
            client_socket, addr = server.accept()
            log_message(f"Accepted connection from {addr}")

            # Handle the gym client in the current thread to maintain continuous communication
            handle_gym_client(client_socket)

        except Exception as e:
            log_message(f"Exception in main loop: {e}")
            break

if __name__ == "__main__":
    main()
