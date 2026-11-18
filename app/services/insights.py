"""Dataset insights from processed_data, shown live on https://medicalchatbot.demos.davieldev.com/api"""

from app.domain import symptoms
from app.repositories import processed_data


def condition_names() -> list[str]:
    return processed_data.condition_names()


def top_symptoms(limit: int) -> list[tuple[str, int]]:
    """The symptoms listed by the most conditions, as (symptom, conditions); ties alphabetically."""
    return symptoms.most_shared(processed_data.symptom_blocks(), limit)
