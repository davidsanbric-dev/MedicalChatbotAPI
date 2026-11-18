"""Find the passages that answer a question: a fast vector search, then a careful re-rank.

Steps 2 and 3 on https://medicalchatbot.demos.davieldev.com/api; the "Re-rank scores"
tab (#rerank) charts these numbers for real questions.
"""

import logging

from app.core.embedding import embed_query, relevance_scores
from app.core.exceptions import RetrievalError
from app.domain.passages import RELEVANCE_THRESHOLD, Passage, select_relevant
from app.repositories import documents

logger = logging.getLogger(__name__)

TOP_K = 20  # nearest passages pulled by the vector search; which of them count is app/domain/passages.py


def search(query: str) -> list[Passage]:
    """The best passages for `query`, most relevant first; empty when none is relevant."""
    ranked = rank(query)
    selected = select_relevant(ranked)
    _log(ranked, selected)
    return selected


def rank(query: str) -> list[Passage]:
    """All TOP_K vector-search candidates, re-ranked by the cross-encoder (best first), before
    the threshold and the TOP_N cut. The evaluation (evals/retrieval.py) scores each stage from this."""
    try:
        candidates = documents.nearest(embed_query(query), TOP_K)
        scores = relevance_scores(query, [candidate.content for candidate in candidates])
    except Exception as error:
        raise RetrievalError("Couldn't search the knowledge base. Please try again in a moment.") from error

    return sorted(
        (
            Passage(candidate.content, candidate.similarity or 0.0, score, candidate.condition)
            for candidate, score in zip(candidates, scores, strict=True)
        ),
        key=lambda passage: passage.relevance,
        reverse=True,
    )


def _log(ranked: list[Passage], selected: list[Passage]) -> None:
    for passage in ranked:
        logger.debug("Candidate (similarity %.4f, cross-encoder %.4f): %s", passage.similarity, passage.relevance, passage.content)
    logger.info(
        "Re-ranked %d passages; %d above %s, %d selected",
        len(ranked),
        sum(passage.is_relevant() for passage in ranked),
        RELEVANCE_THRESHOLD,
        len(selected),
    )
    for number, passage in enumerate(selected, 1):
        preview = passage.content if len(passage.content) <= 200 else passage.content[:200] + "…"
        logger.info("Selected %d/%d (cross-encoder %.4f): %s", number, len(selected), passage.relevance, preview)
