"""Ranking metrics over one question's results.

Each takes `hits`: for every returned passage, best first, whether it is relevant.
"""

from collections.abc import Sequence


def precision(hits: Sequence[bool]) -> float:
    """Share of the returned passages that are relevant (0 when nothing was returned)."""
    return sum(hits) / len(hits) if hits else 0.0


def recall(hits: Sequence[bool], total_relevant: int) -> float:
    """Share of all relevant passages that were returned. With k results and more than k
    relevant passages the best possible value is k / total_relevant."""
    return sum(hits) / total_relevant if total_relevant else 0.0


def hit_rate(hits: Sequence[bool]) -> float:
    """1 when at least one returned passage is relevant, else 0."""
    return 1.0 if any(hits) else 0.0


def reciprocal_rank(hits: Sequence[bool]) -> float:
    """1 / position of the first relevant passage (1.0 when it is first), 0 when none is."""
    return next((1 / position for position, hit in enumerate(hits, 1) if hit), 0.0)
