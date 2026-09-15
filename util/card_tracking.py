import json
import logging

from sqlalchemy import update
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.exc import SQLAlchemyError

from db.models import CardPicked, CardPerformance
from db.session import session_scope

logger = logging.getLogger(__name__)

_INSERT_BY_DIALECT = {"postgresql": postgresql.insert, "sqlite": sqlite.insert}


def track_card_pick(game_state, action, game_id):
    """
    Track the card picked by the agent and store it in the database.
    """
    if not action.startswith("CHOOSE"):
        return

    try:
        chosen_index = int(action.split()[1])
    except (IndexError, ValueError):
        logger.warning("Could not parse a card index from %r", action)
        return

    cards = game_state["game_state"]["screen_state"].get("cards", [])
    if not 0 <= chosen_index < len(cards):
        logger.warning("Chosen card index %s is out of range for %s cards", chosen_index, len(cards))
        return

    chosen_card = cards[chosen_index]
    other_options = [card["name"] for i, card in enumerate(cards) if i != chosen_index]

    try:
        with session_scope() as db:
            db.add(CardPicked(
                game_id=game_id,
                card_name=chosen_card["name"],
                card_id=chosen_card["id"],
                other_options=json.dumps(other_options),
                agent_class=game_state["game_state"].get("class"),
            ))
    except SQLAlchemyError:
        logger.exception("Could not record the pick of %s in game %s", chosen_card["name"], game_id)
        return

    logger.info("Recorded the pick of %s over %s in game %s", chosen_card["name"], other_options, game_id)


def track_card_performance(game_state, floor_reached, won):
    """
    Track the performance of cards in the deck at the end of each game.
    :param game_state: The game state containing the deck information.
    :param floor_reached: The floor the agent reached in this game.
    :param won: Boolean indicating whether the game was won or lost.

    Win rate is stored as a fraction between 0 and 1.
    """
    card_counts = {}
    for card in game_state.get("deck", []):
        if card["name"] in ["Strike", "Defend"]:
            continue
        name, count = card_counts.get(card["id"], (card["name"], 0))
        card_counts[card["id"]] = (name, count + 1)

    try:
        with session_scope() as db:
            insert = _INSERT_BY_DIALECT[db.get_bind().dialect.name]
            games_featured = CardPerformance.games_featured_in
            # Sorted so concurrent writers lock rows in the same order and cannot deadlock.
            for card_id in sorted(card_counts):
                name, count = card_counts[card_id]
                # Insert-if-missing then an in-place UPDATE, so concurrent writers never lose a count.
                db.execute(
                    insert(CardPerformance)
                    .values(card_id=card_id, card_name=name, times_picked=0,
                            average_floor_reached=0.0, win_rate=0.0, games_featured_in=0)
                    .on_conflict_do_nothing(index_elements=[CardPerformance.card_id])
                )
                db.execute(
                    update(CardPerformance)
                    .where(CardPerformance.card_id == card_id)
                    .values(
                        times_picked=CardPerformance.times_picked + count,
                        games_featured_in=games_featured + 1,
                        average_floor_reached=(CardPerformance.average_floor_reached * games_featured + floor_reached) / (games_featured + 1),
                        win_rate=(CardPerformance.win_rate * games_featured + (1.0 if won else 0.0)) / (games_featured + 1),
                    )
                )
    except SQLAlchemyError:
        logger.exception("Could not update card performance")
