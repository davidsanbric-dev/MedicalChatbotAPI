from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder


# Load the embedding model once at module level.
# MUST match the worker's passage model (medical_conditions_worker
# DEFAULT_EMBEDDING_MODEL) or the query and corpus vectors live in different
# spaces and similarity is meaningless. For bge-small-en-v1.5, fastembed's
# query_embed() and passage_embed() yield identical vectors, so the two sides
# stay comparable.
query_embedding_model = TextEmbedding(model_name='BAAI/bge-small-en-v1.5')
cross_encoder_model = TextCrossEncoder(model_name='Xenova/ms-marco-MiniLM-L-6-v2')

# Function to generate embeddings (vectors) for a given query text. Returns a list of floats.
def encode_to_embedding(query: str) -> list[float]:
    return list(query_embedding_model.query_embed([query]))[0]

def generate_relevance_scores(reranking_pairs: list[tuple[str, str]]) -> list[float]:
    return list(cross_encoder_model.rerank_pairs(reranking_pairs))