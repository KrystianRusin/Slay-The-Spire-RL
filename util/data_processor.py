import logging

from util.boss_tracking import update_boss_count
from util.card_tracking import track_card_pick
from util.class_tracking import record_game_start

logger = logging.getLogger(__name__)


def process_game_state(game_state, action, game_id):
    """Record what the action taken on this game state means for the database.

    Returns the ID of the game in progress: a new one when the action starts a
    game, otherwise the game_id passed in. Pass it back in on the next step.
    """
    if action.startswith("START"):
        return record_game_start(action)

    screen_type = game_state.get("game_state", {}).get("screen_type")
    if screen_type not in ("CARD_REWARD", "BOSS_REWARD"):
        return game_id

    if game_id is None:
        logger.warning("No game has been recorded yet; not tracking %s on %s", action, screen_type)
        return None

    if screen_type == "CARD_REWARD":
        track_card_pick(game_state, action, game_id)
    elif action.startswith("CHOOSE"):
        # Taking the relic closes the screen, so this counts each boss once.
        update_boss_count(game_id)

    return game_id
