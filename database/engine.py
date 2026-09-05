"""
database/engine.py
-------------------
SQLAlchemy async engine and session factory for SQLite.

Why async?
----------
FastAPI is an async framework. Using an async engine (via aiosqlite)
means database I/O is non-blocking and won't stall the event loop
during concurrent scan runs.

The module exposes:
  - `engine`              — The shared AsyncEngine instance
  - `AsyncSessionFactory` — An async_sessionmaker bound to the engine
  - `get_async_session()` — An async context manager for dependency injection
  - `init_db()`           — Creates all ORM tables (called once at startup)
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from utils.exceptions import DatabaseError
from utils.logging_config import get_logger

logger = get_logger(__name__)


def _build_engine(database_url: str) -> AsyncEngine:
    """
    Construct the SQLAlchemy AsyncEngine with SQLite-appropriate settings.

    Parameters
    ----------
    database_url : str
        An async-compatible SQLAlchemy URL, e.g.
        ``sqlite+aiosqlite:///data/compliance_watch.db``
    """
    # Ensure the parent directory for the SQLite file exists
    if database_url.startswith("sqlite"):
        # Extract file path from URL: sqlite+aiosqlite:///path/to/file.db
        db_path_str = database_url.split("///", maxsplit=1)[-1]
        if db_path_str and db_path_str != ":memory:":
            db_path = Path(db_path_str)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            logger.debug("SQLite database directory ensured", extra={"path": str(db_path.parent)})

    return create_async_engine(
        database_url,
        echo=False,               # Set True temporarily to debug SQL
        pool_pre_ping=True,       # Re-check connections before use
        connect_args={
            "check_same_thread": False,  # Required for SQLite with async
        },
    )


# ---------------------------------------------------------------------------
# Module-level singletons — created lazily on first import of this module
# ---------------------------------------------------------------------------

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def _get_or_create_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        from config.settings import get_settings
        settings = get_settings()
        _engine = _build_engine(settings.database_url)
        logger.info("SQLAlchemy async engine created", extra={"url": settings.database_url})
    return _engine


def _get_or_create_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        engine = _get_or_create_engine()
        _session_factory = async_sessionmaker(
            bind=engine,
            class_=AsyncSession,
            expire_on_commit=False,  # Don't lazily load after commit (async-unsafe)
            autocommit=False,
            autoflush=False,
        )
    return _session_factory


@property
def engine() -> AsyncEngine:
    return _get_or_create_engine()


# Public alias so other modules can do: from database.engine import AsyncSessionFactory
AsyncSessionFactory = _get_or_create_session_factory


@asynccontextmanager
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager that yields a database session and guarantees
    commit-on-success / rollback-on-exception semantics.

    Usage
    -----
    >>> async with get_async_session() as session:
    ...     result = await session.execute(select(Merchant))
    ...     merchants = result.scalars().all()
    """
    factory = _get_or_create_session_factory()
    session: AsyncSession = factory()
    try:
        yield session
        await session.commit()
        logger.debug("Database session committed successfully.")
    except Exception as exc:
        await session.rollback()
        logger.error(
            "Database session rolled back due to exception.",
            exc_info=True,
            extra={"error_type": type(exc).__name__},
        )
        raise DatabaseError(
            message=f"Database operation failed: {exc}",
            operation="session_commit",
        ) from exc
    finally:
        await session.close()


async def init_db() -> None:
    """
    Create all database tables defined in database.models.

    This is safe to call multiple times — SQLAlchemy's `create_all` with
    `checkfirst=True` (the default) is idempotent.

    Should be called once during application startup (in FastAPI lifespan).
    """
    # Import here to avoid circular imports at module load time
    from database import models  # noqa: F401  — ensures all models are registered

    db_engine = _get_or_create_engine()
    async with db_engine.begin() as conn:
        from database.models import Base
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables initialised (create_all complete).")
