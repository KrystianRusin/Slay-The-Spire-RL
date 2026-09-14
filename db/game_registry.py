"""Which game instances are running, where each one's middleman listens, and which actor owns each.

Middlemen register and keep re-registering while they run; actors claim a game
and keep renewing the claim. Either one that stops reporting for
EXPIRY_SECONDS is treated as gone. Why assignment works this way is recorded
in docs/adr/0005.
"""

import logging
import time
from datetime import datetime, timedelta, timezone

from sqlalchemy import case, delete, or_, select, update

from db.models import GameInstance
from db.session import session_scope

HEARTBEAT_SECONDS = 10
EXPIRY_SECONDS = 60

logger = logging.getLogger(__name__)


class ClaimLost(RuntimeError):
    """The actor's claim on its game expired and passed to another actor, or was released."""


def utc_now():
    return datetime.now(timezone.utc).replace(tzinfo=None)


class GameRegistry:
    """Game registrations and actor claims, kept in the database so every process sees the same assignment."""

    def __init__(self, clock=utc_now):
        self._clock = clock
        self._expiry = timedelta(seconds=EXPIRY_SECONDS)

    def register(self, game_id, host, port):
        """Record that game_id's middleman is running and listening at host:port, keeping any claim on the game."""
        with session_scope() as session:
            game = session.get(GameInstance, game_id)
            if game is None:
                session.add(GameInstance(game_id=game_id, host=host, port=port, middleman_seen_at=self._clock()))
            else:
                game.host, game.port, game.middleman_seen_at = host, port, self._clock()

    def deregister(self, game_id, port):
        """Remove game_id, unless a newer middleman has since registered it on another port."""
        with session_scope() as session:
            session.execute(delete(GameInstance).where(GameInstance.game_id == game_id, GameInstance.port == port))

    def claim(self, actor_id):
        """Claim a running game no other active actor owns, and return its id, or None if there is none.

        A game actor_id already owns is preferred, so a restarted actor goes
        back to the game it was playing.
        """
        now = self._clock()
        claimable = (GameInstance.middleman_seen_at > now - self._expiry) & or_(
            GameInstance.actor_id.is_(None),
            GameInstance.actor_id == actor_id,
            GameInstance.actor_seen_at <= now - self._expiry,
        )
        with session_scope() as session:
            candidates = session.scalars(
                select(GameInstance.game_id)
                .where(claimable)
                .order_by(case((GameInstance.actor_id == actor_id, 0), else_=1), GameInstance.game_id)
            ).all()
        for game_id in candidates:
            # The same conditions again, so a game another actor claimed since the select is left alone.
            with session_scope() as session:
                claimed = session.execute(
                    update(GameInstance)
                    .where(GameInstance.game_id == game_id, claimable)
                    .values(actor_id=actor_id, actor_seen_at=now)
                ).rowcount
            if claimed:
                return game_id
        return None

    def renew(self, game_id, actor_id):
        """Keep actor_id's claim on game_id from expiring. Raises ClaimLost if actor_id no longer owns it."""
        with session_scope() as session:
            renewed = session.execute(
                update(GameInstance)
                .where(GameInstance.game_id == game_id, GameInstance.actor_id == actor_id)
                .values(actor_seen_at=self._clock())
            ).rowcount
        if not renewed:
            raise ClaimLost(f"Actor {actor_id} no longer owns game {game_id}")

    def release(self, game_id, actor_id):
        """Give up actor_id's claim on game_id, if it still owns it."""
        with session_scope() as session:
            session.execute(
                update(GameInstance)
                .where(GameInstance.game_id == game_id, GameInstance.actor_id == actor_id)
                .values(actor_id=None, actor_seen_at=None)
            )

    def address(self, game_id, actor_id):
        """The (host, port) game_id's middleman listens on, or None while no running middleman is registered for it.

        Raises ClaimLost if another actor owns the game now.
        """
        with session_scope() as session:
            game = session.get(GameInstance, game_id)
            if game is None:
                return None
            if game.actor_id != actor_id:
                raise ClaimLost(f"Actor {actor_id} no longer owns game {game_id}")
            if game.middleman_seen_at <= self._clock() - self._expiry:
                return None
            return game.host, game.port


def wait_for_game(registry, actor_id, poll_seconds=HEARTBEAT_SECONDS, sleep=time.sleep):
    """Block until actor_id claims a game, and return its id."""
    while True:
        game_id = registry.claim(actor_id)
        if game_id is not None:
            return game_id
        logger.info("Actor %s is waiting for an unclaimed game to register", actor_id)
        sleep(poll_seconds)
