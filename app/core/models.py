"""The pipeline's tables. This service owns the schema and creates it at startup
(app/core/database.py); the scraper and the worker keep mirrors of the tables they write.

All tables and how they link: https://medicalchatbot.demos.davieldev.com/database#schema
"""

import uuid
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, Index, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class RawData(Base):
    """One scraped page, stored exactly as parsed (written by the scraper)."""

    __tablename__ = "raw_data"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    content: Mapped[dict[str, Any]] = mapped_column(JSONB)
    source: Mapped[str]  # "medical_conditions:<slug>"
    created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ProcessedData(Base):
    """The clean version of one page, tagged with the worker version that made it."""

    __tablename__ = "processed_data"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    raw_data_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("raw_data.id"))
    content: Mapped[dict[str, Any]] = mapped_column(JSONB)
    processor_version: Mapped[str]
    condition_name: Mapped[str]
    processing_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Document(Base):
    """One searchable passage and its embedding (written by the worker)."""

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    processed_data_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("processed_data.id", ondelete="SET NULL"))
    content: Mapped[str] = mapped_column(Text)
    content_type: Mapped[str]  # "symptom" | "paragraph" | "list_item"
    metadata_: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    embedding: Mapped[Any] = mapped_column(Vector(384), nullable=True)  # BAAI/bge-small-en-v1.5
    created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # HNSW, not ivfflat: ivfflat learns its clusters from the rows present when it is
    # built, so one created on the empty table is useless until rebuilt, while HNSW
    # stays correct as rows arrive. The opclass matches the cosine search.
    __table_args__ = (
        Index(
            "documents_embedding_idx",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


class ChatHistory(Base):
    """One chat session's LangChain messages (written by this API)."""

    __tablename__ = "chat_history"

    session_id: Mapped[str] = mapped_column(primary_key=True)
    history: Mapped[list[dict[str, Any]]] = mapped_column(JSONB)
    created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
