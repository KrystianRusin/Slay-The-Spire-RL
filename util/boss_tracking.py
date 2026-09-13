import logging

from sqlalchemy import update
from sqlalchemy.exc import SQLAlchemyError

from db.models import Game
from db.session import session_scope

logger = logging.getLogger(__name__)


def update_boss_count(game_id):
    """
    Update the count of bosses defeated in the database.
    """
    try:
        with session_scope() as db:
            result = db.execute(
                update(Game)
                .where(Game.game_id == game_id)
                .values(bosses_defeated=Game.bosses_defeated + 1)
            )
    except SQLAlchemyError:
        logger.exception("Could not update the boss count for game %s", game_id)
        return

    if result.rowcount:
        print("Boss count updated in the database.")
    else:
        logger.warning("Game %s not found; boss count not updated", game_id)
