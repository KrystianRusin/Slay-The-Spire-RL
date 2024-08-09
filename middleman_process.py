import socket
import threading
import sys
import json

def handle_gym_client(gym_client_socket, game_state_json):
    try:
        # Forward the game state to the gym client
        gym_client_socket.sendall(game_state_json.encode('utf-8'))
        
        # Receive the response from the gym client
        response = gym_client_socket.recv(4096)
        if response:
            # Send the response back to the communication mod (game) via stdout
            sys.stdout.write(response.decode('utf-8') + "\n")
            sys.stdout.flush()
    except Exception as e:
        print(f"Exception: {e}")

    gym_client_socket.close()

def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("0.0.0.0", 9999))
    server.listen(5)
    # Send ready signal to the communication mod (game)
    sys.stdout.write("ready\n")
    sys.stdout.flush()

    while True:
        # Read the game state from stdin
        game_state_json = sys.stdin.readline().strip()
        if not game_state_json:
            break

        # Accept a connection from the environment process
        client_socket, addr = server.accept()
        print(f"Accepted connection from {addr}")

        # Handle the gym client in a new thread
        gym_client_handler = threading.Thread(target=handle_gym_client, args=(client_socket, game_state_json))
        gym_client_handler.start()

if __name__ == "__main__":
    main()
