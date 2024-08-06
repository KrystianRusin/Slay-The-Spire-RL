import sys
import json
import time
def main():
     # Send ready signal to the communication mod (game)
    sys.stdout.write("ready\n")
    sys.stdout.flush()
    while True:
        game_state_json = sys.stdin.readline().strip()

        if game_state_json:
            # Generate a unique filename using the current timestamp
            filename = f"game_state_{int(time.time())}.json"
            
           # Write the game state to the file
            with open(filename, 'w') as file:
                json.dump(json.loads(game_state_json), file)



if __name__ == "__main__":
    main()
