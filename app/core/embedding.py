"""The two small models behind retrieval (fastembed, ONNX on CPU): steps 2 and 3 on
https://medicalchatbot.demos.davieldev.com/api
"""

from functools import cache

import numpy as np
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

# Must match the worker's passage model (medical_conditions_worker/services/embedding.py),
# or questions and passages land in different vector spaces. For bge-small-en-v1.5,
# fastembed's query_embed() and passage_embed() yield identical vectors.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


@cache
def _embedder() -> TextEmbedding:
    return TextEmbedding(EMBEDDING_MODEL)


@cache
def _reranker() -> TextCrossEncoder:
    return TextCrossEncoder(RERANK_MODEL)


def embed_query(text: str) -> np.ndarray:
    return next(_embedder().query_embed(text))


def relevance_scores(query: str, passages: list[str]) -> list[float]:
    """The cross-encoder's score of each passage for the query: higher is more relevant."""
    return list(_reranker().rerank(query, passages))


def warm_up() -> None:
    """Load and run both models once, so the first visitor doesn't pay the ONNX warm-up."""
    embed_query("warm-up")
    relevance_scores("warm-up", ["warm-up"])
