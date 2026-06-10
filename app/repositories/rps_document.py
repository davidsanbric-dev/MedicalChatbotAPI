from sqlalchemy import text

from app.db.database import SessionLocal

class DocumentRepository:
    def __init__(self):
        pass

    def find_similar_to_query(self, query_embedding, top_k=10):
        """Search for similar documents using vector similarity"""
        db = SessionLocal()
        try:
            # Convert the embedding list to the proper vector format
            embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
            
            print(f"[DB Query] Requesting top_k={top_k} documents from database")
            
            # Use the PostgreSQL function we created in init-db.sql
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
            
            print(f"[DB Query] Retrieved {len(results)} documents from database")
            
            return results
        finally:
            db.close()