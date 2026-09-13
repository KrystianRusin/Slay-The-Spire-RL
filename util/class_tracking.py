import logging
from datetime import datetime

from sqlalchemy.exc import SQLAlchemyError

from db.models import Game
from db.session import session_scope

logger = logging.getLogger(__name__)


def record_game_start(action):
    """Record a new game for the class a START action picks.

    Returns the game ID the database assigned, or None if the game could not be
    recorded.
    """
    class_name = action.split()[1]
    try:
        with session_scope() as db:
            new_game = Game(agent_class=class_name, start_time=datetime.now())
            db.add(new_game)
            db.flush()
            game_id = new_game.game_id
    except SQLAlchemyError:
        logger.exception("Could not record the start of a %s game", class_name)
        return None

    print(f"Game started with class '{class_name}' and added to the database with game_id: {game_id}.")
    return game_id
