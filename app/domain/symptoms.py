"""Which symptoms the conditions have in common."""

from collections import Counter
from collections.abc import Iterable
from typing import Any


def symptoms_in(block: Any) -> set[str]:
    """The symptoms of one condition's block, [heading, [symptom, ...]]; empty when it is malformed."""
    if not (isinstance(block, list) and len(block) >= 2 and isinstance(block[1], list)):
        return set()
    return {symptom.strip() for symptom in block[1] if isinstance(symptom, str) and symptom.strip()}


def most_shared(blocks: Iterable[Any], limit: int) -> list[tuple[str, int]]:
    """The symptoms listed by the most conditions, as (symptom, conditions); ties alphabetically."""
    counts = Counter(symptom for block in blocks for symptom in symptoms_in(block))
    return sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:limit]
