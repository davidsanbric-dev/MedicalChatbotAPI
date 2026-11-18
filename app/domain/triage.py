"""Emergency triage: wording that must never wait for a model's judgement, and what the
visitor is told when it appears."""

import re

# English and Spanish.
EMERGENCY_PATTERN = re.compile(
    r"chest pain|can'?t breathe|cannot breathe|trouble breathing|struggling to breathe|"
    r"suicid|kill(?:ing)? myself|end(?:ing)? my (?:own )?life|take my (?:own )?life|want to die|"
    r"overdos|unconscious|passed out|"
    r"face (?:is )?drooping|slurred speech|severe bleeding|bleeding (?:heavily|a lot)|"
    r"dolor (?:de|en el) pecho|no puedo respirar|me cuesta respirar|quitarme la vida|"
    r"suicidarme|matarme|sobredosis|inconsciente|sangrado (?:fuerte|abundante)",
    re.IGNORECASE,
)

EMERGENCY_ADVICE = (
    "This may be a medical emergency. Please call your local emergency number now "
    "(123 in Colombia, 911 in the US and Mexico) or go to the nearest emergency room. "
    "I have also alerted our care team (request {id})."
)


def is_emergency(message: str) -> bool:
    return EMERGENCY_PATTERN.search(message) is not None
