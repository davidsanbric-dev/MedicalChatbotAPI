"""processed_data: the clean version of each condition page (written by the worker)."""

from typing import Any

from sqlalchemy import select

from app.core.database import SessionLocal
from app.core.models import ProcessedData


def condition_names() -> list[str]:
    """Every condition in the knowledge base, alphabetically."""
    with SessionLocal() as session:
        query = (select(ProcessedData.condition_name).distinct().order_by(ProcessedData.condition_name))
        return list(session.scalars(query))


def symptom_blocks() -> list[Any]:
    """Each record's symptom block, as stored: [heading, [symptom, ...]] (or anything else on a bad record)."""
    with SessionLocal() as session:
        return list(session.scalars(select(ProcessedData.content["symptoms"])))
