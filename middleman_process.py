import socket
import sys
import json
import time

def log_message(message):
    """Log messages to a file."""
    with open("middleman_log.txt", "a") as log_file:
        log_file.write(message + "\n")

def save_game_state(game_state_json):
    """Save the game state to a file with a timestamp."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"game_state_{timestamp}.json"
    with open(filename, 'w') as f:
        f.write(game_state_json)

def handle_gym_client(gym_client_socket):
    last_game_state = None  # Store the last game state sent to the gym client
    
    while True:
        try:
            # Read the game state from stdin
            game_state_json = sys.stdin.readline().strip()
            log_message(f"Received game state: {game_state_json}")
            if not game_state_json:
                log_message("No game state received, breaking the loop.")
                break

            # Parse the JSON game state
            try:
                game_state = json.loads(game_state_json)
            except json.JSONDecodeError:
                log_message("Received invalid JSON. Waiting for the next update...")
                continue

            # Check if the game state contains an error
            if "error" in game_state:
                log_message(f"Error in game state detected: {game_state['error']}")
                time.sleep(2)  # Wait for 2 seconds
                
                # Issue the 'state' command to the game to get the current valid state
                sys.stdout.write("STATE\n")
                sys.stdout.flush()
                log_message("Sent STATE command to game.")

                # Read the next game state after requesting the state update
                game_state_json = sys.stdin.readline().strip()
                log_message(f"Received game state after STATE command: {game_state_json}")
                try:
                    game_state = json.loads(game_state_json)
                except json.JSONDecodeError:
                    log_message("Failed to decode JSON after state command. Skipping...")
                    continue

                # Check if the new game state is also erroneous
                if "error" in game_state:
                    log_message(f"Error still present after STATE command: {game_state['error']}. Skipping this state...")
                    continue

            # Save the valid game state to a file before forwarding it
            save_game_state(game_state_json)
            # log_message(f"Saved and forwarding valid game state: {game_state_json}")

            # Forward the valid game state to the gym client
            gym_client_socket.sendall(game_state_json.encode('utf-8'))
            last_game_state = game_state  # Update the last sent game state
            log_message("Game state sent to gym client.")
            
            # Receive the response (chosen action) from the gym client
            response = gym_client_socket.recv(4096)
            if response:
                # Print the received command to stdout
                command = response.decode('utf-8')
                log_message(f"Received command from gym client: {command}")
                
                # Send the response back to the communication mod (game) via stdout
                sys.stdout.write(command + "\n")
                sys.stdout.flush()
                log_message(f"Sent command to game: {command}")

        except Exception as e:
            log_message(f"Exception: {e}")
            break

    gym_client_socket.close()

def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("0.0.0.0", 9999))
    server.listen(5)
    # Send ready signal to the communication mod (game)
    sys.stdout.write("ready\n")
    sys.stdout.flush()
    log_message("Middleman process started and ready.")

    while True:
        # Accept a connection from the environment process
        client_socket, addr = server.accept()
        log_message(f"Accepted connection from {addr}")

        # Handle the gym client in the current thread to maintain continuous communication
        handle_gym_client(client_socket)

if __name__ == "__main__":
    main()
