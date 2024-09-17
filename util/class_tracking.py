from sqlalchemy.orm import Session
from datetime import datetime
from db.models import Game
from db.session import SessionLocal

def track_favorite_class(action, game_id):
    """
    Track the agent's favorite class and store it in the database.
    """
    if action.startswith("START"):
        try:
            # Extract class name from the action string
            class_name = action.split()[1]  # "IRONCLAD" or "SILENT"
            db: Session = SessionLocal()

            # Insert a new game entry
            new_game = Game(
                agent_class=class_name,
                start_time=datetime.now(),
                floors_reached=0,  # Default value, to be updated later
                bosses_defeated=0,  # Default value, to be updated later
                win=False  # Default value, to be updated later
            )

            db.add(new_game)
            db.commit()

            # Retrieve the generated game_id to pass it back
            game_id = new_game.game_id
            db.close()

            print(f"Game started with class '{class_name}' and added to the database with game_id: {game_id}.")
            return game_id

        except Exception as e:
            print(f"Error while tracking favorite class: {e}")
            return None