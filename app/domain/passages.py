"""A passage from the knowledge base, and which ones are relevant enough to answer from.

The rule is shared by the chat, the agent and the retrieval evaluation (evals/retrieval.py),
which sweeps other thresholds over the same candidates.
"""

from dataclasses import dataclass

# Cross-encoder score; at or below it a passage is off-topic. Chosen with `make eval-retrieval`
# (evals/results/retrieval.md): -3 answers more real questions than -2 while still
# returning nothing for nearly every question the knowledge base cannot answer.
RELEVANCE_THRESHOLD = -3.0
TOP_N = 5  # at most this many reach the LLM


@dataclass(frozen=True)
class Passage:
    content: str
    similarity: float  # vector search: 1 - cosine distance
    relevance: float  # cross-encoder score
    condition: str | None = None  # the condition page it came from

    def is_relevant(self, threshold: float = RELEVANCE_THRESHOLD) -> bool:
        return self.relevance > threshold


def select_relevant(ranked: list[Passage], threshold: float = RELEVANCE_THRESHOLD) -> list[Passage]:
    """The passages to answer from: the relevant ones of `ranked` (best first), at most TOP_N.
    Empty when none is relevant, which turns an off-topic question into an explicit "nothing found"."""
    return [passage for passage in ranked if passage.is_relevant(threshold)][:TOP_N]
