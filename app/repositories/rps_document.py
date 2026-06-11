import logging

from app.core.database import session_scope
from app.core.models import Document

logger = logging.getLogger(__name__)


class DocumentRepository:
    def find_similar_to_query(self, query_embedding, top_k=20):
        """Search for the documents whose embeddings are nearest to the query.

        Uses pgvector's cosine distance operator via the ORM, so similarity_score
        is `1 - cosine_distance` (1.0 = identical, 0.0 = orthogonal).
        """
        logger.debug("Requesting top_k=%d documents from database", top_k)

        distance = Document.embedding.cosine_distance(query_embedding)

        with session_scope() as db:
            rows = (
                db.query(
                    Document.id,
                    Document.content,
                    Document.metadata_,
                    distance.label("distance"),
                )
                .order_by(distance)
                .limit(top_k)
                .all()
            )

        results = [
            {
                'id': str(row.id),
                'content': row.content,
                'metadata_': row.metadata_,
                'similarity_score': 1 - float(row.distance) if row.distance is not None else 0.0,
            }
            for row in rows
        ]

        logger.debug("Retrieved %d documents from database", len(results))

        return results