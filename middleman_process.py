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




# Connect to the middleman server
    # middleman_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # middleman_socket.connect(("127.0.0.1", 9999))

    # while True:
    #     # Wait for the game state from the middleman
    #     game_state_json = middleman_socket.recv(4096).decode('utf-8')

    #     game_state = json.loads(game_state_json)
    #     print("Received game state:")
    #     print(json.dumps(game_state, indent=4))

    #     # Create the environment with the received game state
    #     env = SlayTheSpireEnv(game_state)
        
    #     # Reset the environment
    #     state = env.reset()

    #     # while True:
    #     #     # Prompt the user to enter an action
    #     #     command = input("Enter command: ")
    #     #     action = {'command': command}
            
    #     #     if command == 'start':
    #     #         action['PlayerClass'] = input("Enter PlayerClass: ")
    #     #         action['AscensionLevel'] = int(input("Enter AscensionLevel: "))
    #     #         action['Seed'] = input("Enter Seed: ")
    #     #     elif command == 'potion':
    #     #         action['UseDiscard'] = input("Enter Use or Discard: ")
    #     #         action['PotionSlot'] = int(input("Enter PotionSlot: "))
    #     #         action['TargetIndex'] = int(input("Enter TargetIndex: "))
    #     #     elif command == 'play':
    #     #         action['CardIndex'] = int(input("Enter CardIndex: "))
    #     #         if input("Does this card have a target? (yes/no): ") == 'yes':
    #     #             action['TargetIndex'] = int(input("Enter TargetIndex: "))
    #     #     elif command == 'click':
    #     #         action['LeftRight'] = input("Enter Left or Right: ")
    #     #         action['X'] = int(input("Enter X coordinate: "))
    #     #         action['Y'] = int(input("Enter Y coordinate: "))
    #     #     elif command == 'wait':
    #     #         action['Timeout'] = int(input("Enter Timeout: "))

    #     #     # Perform a step in the environment
    #     #     state, reward, done, info = env.step(action)
            
    #     #     print("State after action:")
    #     #     print(json.dumps(state, indent=4))
    #     #     print("Reward:", reward, "Done:", done)
    #     #     print("Info:", info)

    #     #     if done:
    #     #         print("Game is done. Resetting environment.")
    #     #         break

    #     # # Send the updated state back to the middleman server
    #     # response = {
    #     #     "state": state,
    #     #     "reward": reward,
    #     #     "done": done,
    #     #     "info": info
    #     # }
    #     # middleman_socket.sendall(json.dumps(response).encode('utf-8'))