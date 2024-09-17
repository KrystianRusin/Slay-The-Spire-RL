from sqlalchemy.orm import Session
from datetime import datetime
from db.models import Game
from db.session import SessionLocal

def update_game_stats_on_game_over(game_state, game_id):
    """
    Update the game stats in the database when the game is over.
    """
    if game_state.get("game_state", {}).get("screen_type") == "GAME_OVER":
        try:
            db: Session = SessionLocal()

            game = db.query(Game).filter(Game.game_id == game_id).first()
            if game:
                # Update game stats
                game.end_time = datetime.now()
                game.floors_reached = game_state["game_state"].get("floor", 0)
                game.win = game_state["game_state"]["screen_state"].get("victory", False)
                
                # Check if any boss was defeated during the run
                if "BOSS_REWARD" in game_state.get("game_state", {}).get("screen_type"):
                    game.bosses_defeated += 1

                db.commit()
                print(f"Game {game_id} stats updated in the database.")
            else:
                print(f"Game with ID {game_id} not found.")

            db.close()

        except Exception as e:
            print(f"Error while updating game stats: {e}")
