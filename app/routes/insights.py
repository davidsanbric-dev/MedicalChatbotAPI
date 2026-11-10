"""Dataset insights, read live by https://medicalchatbot.demos.davieldev.com/api"""

from typing import Annotated

from fastapi import APIRouter, Query

from app.services import insights

router = APIRouter()

# Plain `def` handlers: FastAPI runs these blocking database reads in its thread pool.


@router.get("/condition-names")
def condition_names() -> list[str]:
    """Every condition in the knowledge base, alphabetically."""
    return insights.condition_names()


@router.get("/top-symptoms")
def top_symptoms(top_n: Annotated[int, Query(ge=1, le=100)] = 10) -> list[tuple[str, int]]:
    """The symptoms shared by the most conditions, as [symptom, number of conditions]."""
    return insights.top_symptoms(top_n)
