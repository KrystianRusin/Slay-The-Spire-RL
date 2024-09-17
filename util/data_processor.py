from util.card_tracking import track_card_pick
from util.class_tracking import track_favorite_class
from sqlalchemy.orm import Session
from db.session import SessionLocal
from db.models import Game

def process_game_state(game_state, action, game_id):
    """
    Central function to process the game state and call the appropriate utility functions.
    """
    screen_type = game_state.get("game_state", {}).get("screen_type")

    # Check for card reward screen type
    if screen_type == "CARD_REWARD":
        track_card_pick(game_state, action, game_id)
    
    # Check for game start action
    if action.startswith("START"):
        track_favorite_class(action, game_id)

def get_next_game_id():
    """
    Get the next available game ID by finding the maximum current game ID and incrementing it.
    """
    try:
        db: Session = SessionLocal()
        # Get the maximum game_id from the games table
        max_game_entry = db.query(Game).order_by(Game.game_id.desc()).first()
        next_game_id = max_game_entry.game_id + 1 if max_game_entry else 1  # Start from 1 if no games exist yet
        db.close()
        return next_game_id
    except Exception as e:
        print(f"Error while fetching the next game ID: {e}")
        return None