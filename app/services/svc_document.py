from app.core.embedding import encode_to_embedding, generate_relevance_scores
from app.schemas import RetrievalRequest
from app.core.exceptions import RetrievalError
from app.repositories.rps_document import DocumentRepository

class DocumentService:
    def __init__(self):
        self.document_repo = DocumentRepository()

    def get_similar_documents(self, payload: RetrievalRequest):
        try:
            query_embedding = encode_to_embedding(payload.query)

            results = self.document_repo.find_similar_to_query(query_embedding)

            print(f"--- Retrieved {len(results)} Chunks from Database ---")
            print("--- Retrieved Chunks Content ---")
            for i, chunk in enumerate(results):
                print(f"--- Chunk {i + 1} ---")
                print(f"Vector Similarity Score: {chunk.get('similarity_score', 'N/A'):.4f}")
                print(chunk['content'])
                print("\n")

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

        print("--- Re-ranked Documents ---")
        print(f"Total re-ranked: {len(reranked_chunks)}, Relevant (score > {RELEVANCE_THRESHOLD}): {len(relevant_chunks)}, Selected: {len(top_documents)}")
        for chunk in top_documents:
            print("--- Chunk ---")
            print(f"Cross-Encoder Score: {chunk['relevance_score']:.4f}")
            print(chunk['content'])
            print("\n")

        return top_documents
    
