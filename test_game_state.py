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
            current_screen_state = game_state.get("game_state", {}).get("screen_state")
            if current_screen_state is not None:
                if current_screen_state != previous_screen_state:
                    screen_state_name = game_state.get("game_state", {}).get("screen_type", "unknown")
                    timestamp = int(time.time())
                    filename = f"game_state_{screen_state_name}_{timestamp}.json"
                    
                    with open(filename, 'w') as file:
                        json.dump(game_state, file, indent=4)

                    previous_screen_state = current_screen_state

if __name__ == "__main__":
    main()