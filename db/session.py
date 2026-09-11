import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.models import Base

load_dotenv()

# The engine and session factory are built lazily rather than at import time.
# Connecting (or creating tables) while the module is being imported makes every
# module that transitively touches the database unimportable without a live,
# correctly-credentialed Postgres - which breaks unit tests and any container
# that starts before the database is healthy.
_engine = None
_session_factory = None


def get_engine():
    """Return the process-wide engine, creating it on first use."""
    global _engine
    if _engine is None:
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise RuntimeError(
                "DATABASE_URL is not set. Copy .env.example to .env and fill it in."
            )
        _engine = create_engine(database_url)
    return _engine


def init_db():
    """Create any missing tables.

    Call this explicitly from an entry point. It used to run as an import-time
    side effect of this module.
    """
    Base.metadata.create_all(get_engine())


def SessionLocal():
    """Return a new database session.

    Named as it is because call sites use it like the sessionmaker it replaced.
    """
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(bind=get_engine())
    return _session_factory()
