"""Database connection, plus the schema setup that runs at startup.

The API owns the pipeline's schema (app/core/models.py); the scraper and the worker
only write to it.
"""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.core.config import DATABASE_URL
from app.core.models import Base

engine = create_engine(DATABASE_URL, pool_size=10, max_overflow=20, pool_pre_ping=True)
SessionLocal = sessionmaker(engine)


def prepare_database() -> None:
    """Create whatever is missing: the vector extension, the tables and their indexes."""
    with engine.begin() as connection:
        # Before create_all: the documents.embedding column type comes from this extension.
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)
