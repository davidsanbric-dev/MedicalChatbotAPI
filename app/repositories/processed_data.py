"""processed_data: the clean version of each condition page (written by the worker)."""

from typing import Any

from sqlalchemy import select

from app.core.models import ProcessedData
from app.repositories.base import BaseRepository


class ProcessedDataRepository(BaseRepository):
    model = ProcessedData

    def condition_names(self) -> list[str]:
        """Every condition in the knowledge base, alphabetically."""
        query = select(ProcessedData.condition_name).distinct().order_by(ProcessedData.condition_name)
        return self._scalars(query)

    def symptom_blocks(self) -> list[Any]:
        """Each record's symptom block, as stored: [heading, [symptom, ...]] (or anything else on a bad record)."""
        return self._scalars(select(ProcessedData.content["symptoms"]))

    def processor_versions(self) -> list[str]:
        """The worker versions that produced the records, oldest first."""
        query = select(ProcessedData.processor_version).distinct().order_by(ProcessedData.processor_version)
        return self._scalars(query)
