from sqlalchemy.orm import Session
from datetime import datetime
from db.models import Game
from db.session import SessionLocal

def update_game_stats_on_game_over(game_state, game_id):
    """
    Update the game stats in the database when the game is over.
    """
    try:
        # Check if 'game_state' and its required properties exist
        game_state_data = game_state.get("game_state", {})
        if not game_state_data:
            print("Error: 'game_state' is missing in the provided game state.")
            return

        if game_state_data.get("screen_type") != "GAME_OVER":
            print("Error: Screen type is not 'GAME_OVER'. No update is required.")
            return

        db: Session = SessionLocal()

        game = db.query(Game).filter(Game.game_id == game_id).first()
        if game:
            # Update game stats
            game.end_time = datetime.now()
            game.floors_reached = game_state_data.get("floor", 0)
            game.win = game_state_data.get("screen_state", {}).get("victory", False)

            db.commit()
            print(f"Game {game_id} stats updated in the database.")
        else:
            print(f"Game with ID {game_id} not found.")

        db.close()

    except Exception as e:
        print(f"Error while updating game stats: {e}")
