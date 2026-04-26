"""Database connection management for RareGeneAI.

Supports PostgreSQL for production and SQLite for development/testing.
Connection URL is configured via environment variable.
"""

from __future__ import annotations

import os
from collections.abc import Generator

from pydantic import BaseModel, Field
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from raregeneai.database.models import Base

# ── Configuration ────────────────────────────────────────────────────────────

_DEFAULT_DB_URL = "sqlite:///raregeneai.db"


class DatabaseConfig(BaseModel):
    """Database configuration."""
    database_url: str = Field(
        default_factory=lambda: os.environ.get("RAREGENEAI_DATABASE_URL", _DEFAULT_DB_URL),
        description="PostgreSQL connection URL (postgresql://user:pass@host:5432/dbname)",
    )
    pool_size: int = Field(default=10, description="Connection pool size")
    max_overflow: int = Field(default=20, description="Max overflow connections")
    pool_timeout: int = Field(default=30, description="Pool timeout in seconds")
    echo_sql: bool = Field(default=False, description="Echo SQL statements (debug)")


# ── Engine and session ───────────────────────────────────────────────────────

_engine = None
_SessionLocal = None


def init_db(config: DatabaseConfig | None = None) -> None:
    """Initialize the database engine and create tables."""
    global _engine, _SessionLocal

    if config is None:
        config = DatabaseConfig()

    db_url = config.database_url

    if db_url.startswith("sqlite"):
        _engine = create_engine(
            db_url,
            connect_args={"check_same_thread": False},
            echo=config.echo_sql,
        )
        # Enable WAL mode for better concurrent access
        @event.listens_for(_engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
    else:
        _engine = create_engine(
            db_url,
            pool_size=config.pool_size,
            max_overflow=config.max_overflow,
            pool_timeout=config.pool_timeout,
            pool_pre_ping=True,
            echo=config.echo_sql,
        )

    _SessionLocal = sessionmaker(bind=_engine, autocommit=False, autoflush=False)

    # Run Alembic migrations to create/update tables.
    # Falls back to create_all for test environments without Alembic config.
    _run_migrations()


def _run_migrations() -> None:
    """Run Alembic migrations if applicable, else fall back to create_all.

    Uses Alembic for file-based databases with alembic.ini present.
    Falls back to create_all for in-memory DBs and test environments.
    """
    from pathlib import Path

    db_url = str(_engine.url)

    # Skip Alembic for in-memory databases (tests)
    if ":memory:" in db_url:
        Base.metadata.create_all(bind=_engine)
        return

    alembic_ini = Path(__file__).resolve().parents[2] / "alembic.ini"
    if alembic_ini.exists():
        try:
            from alembic import command
            from alembic.config import Config

            alembic_cfg = Config(str(alembic_ini))
            alembic_cfg.set_main_option("sqlalchemy.url", db_url)
            command.upgrade(alembic_cfg, "head")
            return
        except Exception as e:
            from loguru import logger
            logger.warning(f"Alembic migration failed, falling back to create_all: {e}")

    # Fallback for tests and environments without Alembic
    Base.metadata.create_all(bind=_engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency: yield a database session."""
    if _SessionLocal is None:
        init_db()
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_session() -> Session:
    """Get a database session for non-FastAPI use."""
    if _SessionLocal is None:
        init_db()
    return _SessionLocal()
