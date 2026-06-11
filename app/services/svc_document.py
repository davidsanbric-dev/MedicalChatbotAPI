import logging

from app.core.embedding import encode_to_embedding, generate_relevance_scores
from app.schemas import RetrievalRequest
from app.core.exceptions import RetrievalError
from app.repositories.rps_document import DocumentRepository

logger = logging.getLogger(__name__)


class DocumentService:
    def __init__(self):
        self.document_repo = DocumentRepository()

    def get_similar_documents(self, payload: RetrievalRequest):
        try:
            query_embedding = encode_to_embedding(payload.query)

            results = self.document_repo.find_similar_to_query(query_embedding)

            logger.info("Retrieved %d chunks from database", len(results))
            for i, chunk in enumerate(results):
                logger.debug(
                    "Chunk %d (similarity=%.4f): %s",
                    i + 1,
                    chunk.get('similarity_score', float('nan')),
                    chunk['content'],
                )

            return {"results": results}

        except Exception as e:
            raise RetrievalError(f"Failed to retrieve chunks: {str(e)}")
        
    @staticmethod
    def set_reranking_pairs(retrieved_documents, rephrased_query):
        reranking_pairs = []
        for chunk in retrieved_documents['results']:
            reranking_pairs.append([rephrased_query, chunk['content']])

        return reranking_pairs
    
    def rerank_chunks(self, retrieved_documents, rephrased_query):
        reranking_pairs = self.set_reranking_pairs(retrieved_documents, rephrased_query)

        relevance_scores = generate_relevance_scores(reranking_pairs)

        for i in range(len(retrieved_documents['results'])):
            retrieved_documents['results'][i]['relevance_score'] = relevance_scores[i]

        reranked_chunks = sorted(retrieved_documents['results'], key=lambda x: x['relevance_score'], reverse=True)
        
        return reranked_chunks
    
    def filter_top_documents(self, retrieved_documents, rephrased_query, top_n):
        reranked_chunks = self.rerank_chunks(retrieved_documents, rephrased_query)

        # Define relevance threshold - cross-encoder scores below this are considered irrelevant
        RELEVANCE_THRESHOLD = -2.0

        # Filter chunks that meet the relevance threshold
        relevant_chunks = [chunk for chunk in reranked_chunks if chunk['relevance_score'] > RELEVANCE_THRESHOLD]

        # Take top N from relevant chunks
        top_documents = relevant_chunks[:top_n]

        logger.info(
            "Re-ranked %d chunks; %d relevant (score > %s), %d selected",
            len(reranked_chunks),
            len(relevant_chunks),
            RELEVANCE_THRESHOLD,
            len(top_documents),
        )
        for chunk in top_documents:
            logger.debug(
                "Selected chunk (cross-encoder=%.4f): %s",
                chunk['relevance_score'],
                chunk['content'],
            )

        return top_documents
    
