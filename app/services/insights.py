"""Dataset insights from processed_data, shown live on https://medicalchatbot.demos.davieldev.com/api"""

from collections import Counter

from sqlalchemy import select

from app.core.database import SessionLocal
from app.core.models import ProcessedData


def condition_names() -> list[str]:
    with SessionLocal() as session:
        query = select(ProcessedData.condition_name).distinct().order_by(ProcessedData.condition_name)
        return list(session.scalars(query))


def top_symptoms(limit: int) -> list[tuple[str, int]]:
    """The symptoms listed by the most conditions, as (symptom, conditions); ties alphabetically."""
    with SessionLocal() as session:
        # Only each record's symptom block, which is [heading, [symptom, ...]]
        symptom_blocks = session.scalars(select(ProcessedData.content["symptoms"])).all()
    counts: Counter[str] = Counter()
    for block in symptom_blocks:
        if isinstance(block, list) and len(block) >= 2 and isinstance(block[1], list):
            counts.update({symptom.strip() for symptom in block[1] if isinstance(symptom, str) and symptom.strip()})
    return sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:limit]
