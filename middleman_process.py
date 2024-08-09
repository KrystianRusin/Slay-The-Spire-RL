import socket
import threading
import sys
import json

def handle_gym_client(gym_client_socket):
    while True:
        try:
            # Read the game state from stdin
            game_state_json = sys.stdin.readline().strip()
            if not game_state_json:
                break

            # Forward the game state to the gym client
            gym_client_socket.sendall(game_state_json.encode('utf-8'))
            
            # Receive the response (chosen action) from the gym client
            response = gym_client_socket.recv(4096)
            if response:
                # Print the received command to stdout
                command = response.decode('utf-8')
                print(command)
                
                # Send the response back to the communication mod (game) via stdout
                sys.stdout.write(command + "\n")
                sys.stdout.flush()
        except Exception as e:
            print(f"Exception: {e}")
            break

    gym_client_socket.close()

def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("0.0.0.0", 9999))
    server.listen(5)
    # Send ready signal to the communication mod (game)
    sys.stdout.write("ready\n")
    sys.stdout.flush()

    while True:
        # Accept a connection from the environment process
        client_socket, addr = server.accept()
        print(f"Accepted connection from {addr}")

        # Handle the gym client in the current thread to maintain continuous communication
        handle_gym_client(client_socket)

if __name__ == "__main__":
    main()
