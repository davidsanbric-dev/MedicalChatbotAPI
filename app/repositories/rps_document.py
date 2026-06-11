import logging

from sqlalchemy import text

from app.core.database import SessionLocal

logger = logging.getLogger(__name__)


class DocumentRepository:
    def __init__(self):
        pass

    def find_similar_to_query(self, query_embedding, top_k=10):
        """Search for similar documents using vector similarity"""
        db = SessionLocal()
        try:
            # Convert the embedding list to the proper vector format
            embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
            
            logger.debug("Requesting top_k=%d documents from database", top_k)
            
            # Use the search_similar_documents() function created via the DDL
            # event listener in app/core/models.py.
            sql_query = text("""
                SELECT id, content, metadata_, similarity_score
                FROM search_similar_documents(:query_embedding, :top_k)
            """)
            
            result = db.execute(sql_query, {
                'query_embedding': embedding_str,
                'top_k': top_k
            })
            
            results = [
                {
                    'id': str(row[0]),
                    'content': row[1],
                    'metadata_': row[2],
                    'similarity_score': float(row[3]) if row[3] is not None else 0.0
                }
                for row in result.fetchall()
            ]
            
            logger.debug("Retrieved %d documents from database", len(results))
            
            return results
        finally:
            db.close()