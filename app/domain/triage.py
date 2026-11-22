"""Emergency triage: wording that must never wait for a model's judgement, and what the
visitor is told when it appears, in their language."""

import re

EMERGENCY_PATTERNS = {
    "en": re.compile(
        r"chest pain|can'?t breathe|cannot breathe|trouble breathing|struggling to breathe|"
        r"suicid|kill(?:ing)? myself|end(?:ing)? my (?:own )?life|take my (?:own )?life|want to die|"
        r"overdos|unconscious|passed out|"
        r"face (?:is )?drooping|slurred speech|severe bleeding|bleeding (?:heavily|a lot)",
        re.IGNORECASE,
    ),
    "es": re.compile(
        r"dolor (?:de|en el) pecho|no puedo respirar|me cuesta respirar|quitarme la vida|"
        r"suicidarme|suicidio|matarme|sobredosis|inconsciente|sangrado (?:fuerte|abundante)",
        re.IGNORECASE,
    ),
}

EMERGENCY_ADVICE = {
    "en": (
        "This may be a medical emergency. Please call your local emergency number now "
        "(123 in Colombia, 911 in the US and Mexico) or go to the nearest emergency room. "
        "I have also alerted our care team (request {id})."
    ),
    "es": (
        "Esto puede ser una emergencia médica. Llama ahora a tu número local de emergencias "
        "(123 en Colombia, 911 en Estados Unidos y México) o ve al servicio de urgencias más cercano. "
        "También avisé a nuestro equipo de atención (solicitud {id})."
    ),
}


def is_emergency(message: str) -> bool:
    return any(pattern.search(message) for pattern in EMERGENCY_PATTERNS.values())


def emergency_advice(message: str, reference: str) -> str:
    """The fixed safety advice (no model is asked), in the language of the emergency wording found."""
    language = "es" if EMERGENCY_PATTERNS["es"].search(message) else "en"
    return EMERGENCY_ADVICE[language].format(id=reference)
