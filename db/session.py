import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.models import Base

load_dotenv()

POOL_OPTIONS = {
    "pool_size": 2,
    "max_overflow": 2,
    "pool_timeout": 30,
    "pool_recycle": 1800,
    "pool_pre_ping": True,
}

# The engine and session factory are built lazily rather than at import time.
# Connecting (or creating tables) while the module is being imported makes every
# module that transitively touches the database unimportable without a live,
# correctly-credentialed Postgres - which breaks unit tests and any container
# that starts before the database is healthy.
_engine = None
_engine_pid = None
_session_factory = None


def get_engine():
    """Return this process's engine, creating it on first use.

    A process forked from one that had already connected builds its own engine
    instead of sharing the parent's pooled connections.
    """
    global _engine, _engine_pid, _session_factory
    if _engine is None or _engine_pid != os.getpid():
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise RuntimeError(
                "DATABASE_URL is not set. Copy .env.example to .env and fill it in."
            )
        if _engine is not None:
            _engine.dispose(close=False)
        _engine = create_engine(database_url, **POOL_OPTIONS)
        _engine_pid = os.getpid()
        _session_factory = sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


def init_db():
    """Create any missing tables.

    Call this explicitly from an entry point. It used to run as an import-time
    side effect of this module.
    """
    Base.metadata.create_all(get_engine())


def _get_session_factory():
    get_engine()
    return _session_factory


def session_scope():
    """Open a session for use in a with block.

    The block runs in one transaction: it commits if the block completes, rolls
    back if it raises, and closes the session either way.
    """
    return _get_session_factory().begin()
