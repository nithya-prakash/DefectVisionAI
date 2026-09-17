"""Database engine/session setup. DATABASE_URL is read from the environment
(see .env.example) so the same code works against Docker Compose's Postgres
service or a local SQLite file for tests."""

import os
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from automotive_vision.database.models import Base

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./automotive_vision.db")

_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=_connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
