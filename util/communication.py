import json
import socket

def handle_end_of_episode(client_socket):
    """
    Handles the end-of-episode scenario by sending the necessary commands
    to navigate through the game over screen and start a new game.
    """
    commands = ["PROCEED", "PROCEED"]

    for command in commands:
        client_socket.sendall(command.encode('utf-8'))
        print(f"Sent '{command}' command")

        try:
            game_state = receive_full_json(client_socket)
            print(f"Game state received after '{command}'")

        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON after '{command}': {e}")
            return
        except ConnectionError as e:
            print(f"Connection error after '{command}': {e}")
            return

def receive_full_json(client_socket):
    data = b''
    while True:
        try:
            part = client_socket.recv(4096)
            if not part:
                raise ConnectionError("Socket connection closed")
            data += part
            try:
                return json.loads(data.decode('utf-8'))
            except json.JSONDecodeError:
                continue
        except socket.timeout:
            print("Timeout occurred while receiving game state. Requesting resend...")
            continue
