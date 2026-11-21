"""Dataset insights from processed_data, shown live on https://medicalchatbot.demos.davieldev.com/api"""

from app.domain import symptoms
from app.repositories.processed_data import ProcessedDataRepository


class InsightsService:
    def __init__(self, processed_data_repository: ProcessedDataRepository) -> None:
        self.processed_data_repository = processed_data_repository

    def condition_names(self) -> list[str]:
        return self.processed_data_repository.condition_names()

    def top_symptoms(self, limit: int) -> list[tuple[str, int]]:
        """The symptoms listed by the most conditions, as (symptom, conditions); ties alphabetically."""
        return symptoms.most_shared(self.processed_data_repository.symptom_blocks(), limit)
