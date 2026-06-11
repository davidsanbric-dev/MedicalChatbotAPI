from fastapi import APIRouter
from typing import List, Tuple
from app.services.svc_processed_data import ProcessedDataService

processed_data_service = ProcessedDataService()

router = APIRouter()

# Service failures (RuntimeError) are mapped to HTTP 500 by the app-level
# exception handlers in app/main.py.

@router.get('/condition-names')
async def get_condition_names() -> List[str]:
    """
    Get all condition names from the scraped JSON files.

    Returns:
        List of condition names (strings) extracted from the "title" field of each JSON file
    """
    return processed_data_service.get_all_condition_names()

@router.get('/top-symptoms')
async def get_top_symptoms(top_n: int = 10) -> List[Tuple[str, int]]:
    """
    Get the top N most common symptoms across all medical conditions.

    Args:
        top_n: Number of top symptoms to return (default: 10)

    Returns:
        List of tuples containing (symptom_name, count_across_conditions)
    """
    return processed_data_service.get_top_symptoms_across_conditions(top_n=top_n)