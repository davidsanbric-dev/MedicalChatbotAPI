from fastapi import APIRouter, HTTPException
from typing import List, Tuple
from app.services.svc_processed_data import ProcessedDataService

processed_data_service = ProcessedDataService()

router = APIRouter()

@router.get('/condition-names')
async def get_condition_names() -> List[str]:
    """
    Get all condition names from the scraped JSON files.

    Returns:
        List of condition names (strings) extracted from the "title" field of each JSON file
    """
    try:
        condition_names = processed_data_service.get_all_condition_names()
        return condition_names

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get('/top-symptoms')
async def get_top_symptoms(top_n: int = 10) -> List[Tuple[str, int]]:
    """
    Get the top N most common symptoms across all medical conditions.

    Args:
        top_n: Number of top symptoms to return (default: 10)

    Returns:
        List of tuples containing (symptom_name, count_across_conditions)
    """
    try:
        top_symptoms = processed_data_service.get_top_symptoms_across_conditions(top_n=top_n)
        return top_symptoms

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")