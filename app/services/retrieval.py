"""Find the passages that answer a question: a fast vector search, then a careful re-rank.

Steps 2 and 3 on https://medicalchatbot.demos.davieldev.com/api; the "Re-rank scores"
tab (#rerank) charts these numbers for real questions.
"""

import logging
from dataclasses import dataclass

from app.core.embedding import embed_query, relevance_scores
from app.core.exceptions import RetrievalError
from app.repositories import documents

logger = logging.getLogger(__name__)

TOP_K = 20  # nearest passages pulled by the vector search
TOP_N = 5  # at most this many reach the LLM
RELEVANCE_THRESHOLD = -2.0  # cross-encoder score; below it a passage is off-topic


@dataclass
class Passage:
    content: str
    similarity: float  # vector search: 1 - cosine distance
    relevance: float  # cross-encoder score


def search(query: str) -> list[Passage]:
    """The best passages for `query`, most relevant first. Empty when none clears the threshold,
    which turns an off-topic question into an explicit "nothing found"."""
    try:
        candidates = documents.nearest(embed_query(query), TOP_K)
        scores = relevance_scores(query, [row.content for row in candidates])
    except Exception as error:
        raise RetrievalError("Couldn't search the knowledge base. Please try again in a moment.") from error

    ranked = sorted(
        (Passage(row.content, row.similarity or 0.0, score) for row, score in zip(candidates, scores, strict=True)),
        key=lambda passage: passage.relevance,
        reverse=True,
    )
    relevant = [passage for passage in ranked if passage.relevance > RELEVANCE_THRESHOLD]
    selected = relevant[:TOP_N]
    _log(ranked, relevant, selected)
    return selected


def _log(ranked: list[Passage], relevant: list[Passage], selected: list[Passage]) -> None:
    for passage in ranked:
        logger.debug("Candidate (similarity %.4f, cross-encoder %.4f): %s", passage.similarity, passage.relevance, passage.content)
    logger.info(
        "Re-ranked %d passages; %d above %s, %d selected", len(ranked), len(relevant), RELEVANCE_THRESHOLD, len(selected)
    )
    for number, passage in enumerate(selected, 1):
        preview = passage.content if len(passage.content) <= 200 else passage.content[:200] + "…"
        logger.info("Selected %d/%d (cross-encoder %.4f): %s", number, len(selected), passage.relevance, preview)
