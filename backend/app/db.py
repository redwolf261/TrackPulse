"""Persistence for TrackPulse prediction history.

Defaults to local SQLite for development (zero-setup, matches how this repo
has always been run locally). In production, set DATABASE_URL to a Postgres
connection string (Render's managed Postgres provides one directly) — the
schema is plain SQLAlchemy with no SQLite-specific features, so the same
models work against either backend unchanged.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import create_engine, String, Float, DateTime, Integer
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

_DEFAULT_SQLITE_PATH = Path(__file__).resolve().parent.parent / "trackpulse.db"


def _resolve_database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        return f"sqlite:///{_DEFAULT_SQLITE_PATH}"
    # Render (and some other hosts) hand out "postgres://" URLs; SQLAlchemy 2.x
    # needs a scheme naming an actual driver. We install psycopg (v3), so route
    # both the bare "postgres://" and "postgresql://" forms to it explicitly
    # rather than relying on SQLAlchemy's default (which assumes psycopg2).
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg://", 1)
    elif url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


DATABASE_URL = _resolve_database_url()
_is_sqlite = DATABASE_URL.startswith("sqlite")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if _is_sqlite else {},
    pool_pre_ping=not _is_sqlite,  # recycle stale connections against a real DB server
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


class Observation(Base):
    """One classified frame/image, belonging to a session (a demo run / upload sequence)."""

    __tablename__ = "observations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    label: Mapped[str] = mapped_column(String)  # DRY | DAMP | WET
    p_dry: Mapped[float] = mapped_column(Float)
    p_damp: Mapped[float] = mapped_column(Float)
    p_wet: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float)  # max class probability
    trend: Mapped[str] = mapped_column(String, default="STABLE")  # WETTING | DRYING | STABLE
    suggestion: Mapped[str] = mapped_column(String, default="")
    image_filename: Mapped[str] = mapped_column(String, default="")


def init_db() -> None:
    Base.metadata.create_all(engine)


def get_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
