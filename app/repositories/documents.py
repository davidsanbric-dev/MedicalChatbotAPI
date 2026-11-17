"""documents: nearest-passage search with pgvector.

The SQL it compiles to is in the "SQL" tab of https://medicalchatbot.demos.davieldev.com/database#sql
"""

from collections.abc import Sequence

import numpy as np
from sqlalchemy import Row, select

from app.core.database import SessionLocal
from app.core.models import Document, ProcessedData


def nearest(query_vector: np.ndarray, limit: int) -> Sequence[Row]:
    """The `limit` passages closest in meaning, closest first, as (content, similarity, condition) rows.

    similarity = 1 - cosine distance: 1.0 points the same way, 0.0 is unrelated.
    condition is the page the passage came from; None once that page's processed row is gone.
    """
    distance = Document.embedding.cosine_distance(query_vector)
    with SessionLocal() as session:
        query = (
            select(Document.content, (1 - distance).label("similarity"), ProcessedData.condition_name.label("condition"))
            .outerjoin(ProcessedData, Document.processed_data_id == ProcessedData.id)
            .order_by(distance)
            .limit(limit)
        )
        return session.execute(query).all()
