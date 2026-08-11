"""SQLite persistence for TrackPulse prediction history."""
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import create_engine, String, Float, DateTime, Integer
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

DB_PATH = Path(__file__).resolve().parent.parent / "trackpulse.db"
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


class Observation(Base):
    """One classified frame/image, belonging to a session (a demo run / upload sequence)."""

    __tablename__ = "observations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
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
