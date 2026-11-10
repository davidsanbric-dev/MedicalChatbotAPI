"""Database connection, plus the schema setup that runs at startup.

The API owns the pipeline's schema (app/core/models.py); the scraper and the worker
only write to it. Why the vector index is HNSW:
https://medicalchatbot.demos.davieldev.com/database#notes
"""

import logging

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.core.config import DATABASE_URL
from app.core.models import Base, Document

logger = logging.getLogger(__name__)

engine = create_engine(DATABASE_URL, pool_size=10, max_overflow=20, pool_pre_ping=True)
SessionLocal = sessionmaker(engine)


def prepare_database() -> None:
    """Create whatever is missing: the vector extension, the tables and their indexes.

    Idempotent, so it runs on every start, and it upgrades databases made by older versions.
    """
    with engine.begin() as connection:
        # Before create_all: the documents.embedding column type comes from this extension.
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        # Leftover of an old schema: an L2-distance search function nothing calls.
        connection.execute(text("DROP FUNCTION IF EXISTS search_similar_documents(vector, integer)"))
    Base.metadata.create_all(engine)
    _replace_outdated_vector_index()


def _replace_outdated_vector_index() -> None:
    """Swap an old ivfflat embedding index for the model's HNSW one.

    create_all only builds indexes for the tables it creates, so databases made before
    the switch keep their ivfflat index: built on an empty table, with the default L2
    opclass the cosine search can't use. Once replaced, this is a no-op.
    """
    index = next(index for index in Document.__table__.indexes if index.name == "documents_embedding_idx")
    with engine.begin() as connection:
        definition = connection.execute(
            text("SELECT indexdef FROM pg_indexes WHERE indexname = :name"), {"name": index.name}
        ).scalar()
        if definition and ("USING hnsw" not in definition or "vector_cosine_ops" not in definition):
            logger.info("Replacing outdated vector index: %s", definition)
            connection.execute(text(f"DROP INDEX {index.name}"))
        index.create(bind=connection, checkfirst=True)
