"""SQLite database (via SQLAlchemy) that stores every prediction."""
import os
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

DB_URL = os.environ.get("DATABASE_URL", "sqlite:///./predictions.db")

engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False)


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True)
    filename: Mapped[str] = mapped_column(String(255))
    label: Mapped[str] = mapped_column(String(10))       # "real" or "fake"
    confidence: Mapped[float] = mapped_column(Float)      # 0.0 - 1.0
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )


def init_db():
    Base.metadata.create_all(engine)