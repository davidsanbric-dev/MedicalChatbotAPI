"""documents: nearest-passage search with pgvector.

The SQL it compiles to is in the "SQL" tab of https://medicalchatbot.demos.davieldev.com/database#sql
"""

from typing import NamedTuple

import numpy as np
from sqlalchemy import select

from app.core.database import SessionLocal
from app.core.models import Document, ProcessedData


class Candidate(NamedTuple):
    """A passage the vector search found, before the cross-encoder scores it."""

    content: str
    similarity: float | None  # 1 - cosine distance: 1.0 points the same way, 0.0 is unrelated
    condition: str | None  # the page it came from; None once that page's processed row is gone


def nearest(query_vector: np.ndarray, limit: int) -> list[Candidate]:
    """The `limit` passages closest in meaning, closest first."""
    distance = Document.embedding.cosine_distance(query_vector)
    with SessionLocal() as session:
        query = (
            select(Document.content, (1 - distance).label("similarity"), ProcessedData.condition_name.label("condition"))
            .outerjoin(ProcessedData, Document.processed_data_id == ProcessedData.id)
            .order_by(distance)
            .limit(limit)
        )
        return [Candidate._make(row) for row in session.execute(query)]
