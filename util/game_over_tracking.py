import logging
from datetime import datetime

from sqlalchemy.exc import SQLAlchemyError

from db.models import Game
from db.session import session_scope

logger = logging.getLogger(__name__)


def update_game_stats_on_game_over(game_state, game_id, total_reward):
    """
    Update the game stats in the database when the game is over.
    """
    game_state_data = game_state.get("game_state", {})
    if not game_state_data:
        logger.warning("No game_state in the payload; game %s stats not updated", game_id)
        return

    if game_state_data.get("screen_type") != "GAME_OVER":
        logger.warning("Screen type is not GAME_OVER; game %s stats not updated", game_id)
        return

    if game_id is None:
        logger.warning("No game has been recorded; game over stats not updated")
        return

    try:
        with session_scope() as db:
            game = db.get(Game, game_id)
            if game:
                game.end_time = datetime.now()
                game.floors_reached = game_state_data.get("floor", 0)
                game.win = game_state_data.get("screen_state", {}).get("victory", False)
                game.reward = total_reward
    except SQLAlchemyError:
        logger.exception("Could not update the stats for game %s", game_id)
        return

    if game:
        logger.info("Recorded the end of game %s", game_id)
    else:
        logger.warning("Game %s not found; stats not updated", game_id)
