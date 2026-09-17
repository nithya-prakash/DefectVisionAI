"""Test-wide setup: points the app at a throwaway SQLite file instead of Postgres,
before any application module (which reads DATABASE_URL at import time) is imported.
"""

import os
from pathlib import Path

TEST_DB_PATH = Path(__file__).parent / "test_automotive_vision.db"
os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DB_PATH}"

import pytest


@pytest.fixture(scope="session", autouse=True)
def _cleanup_test_db_file():
    yield
    TEST_DB_PATH.unlink(missing_ok=True)


@pytest.fixture(autouse=True)
def clean_tables():
    """Drops and recreates all tables before every test for isolation, since the
    SQLite file (and its engine) is shared across the whole test session."""
    from automotive_vision.database.models import Base
    from automotive_vision.database.session import engine

    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield
