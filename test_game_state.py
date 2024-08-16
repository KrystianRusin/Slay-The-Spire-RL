import sys
import json
import time

def main():
    # Send ready signal to the communication mod (game)
    sys.stdout.write("ready\n")
    sys.stdout.flush()

    previous_screen_state = None

    while True:
        game_state_json = sys.stdin.readline().strip()

        if game_state_json:
            game_state = json.loads(game_state_json)
            timestamp = int(time.time())
            filename = f"game_state_{timestamp}.json"
                    
            with open(filename, 'w') as file:
                json.dump(game_state, file, indent=4)

if __name__ == "__main__":
    main()