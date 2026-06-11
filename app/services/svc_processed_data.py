from collections import Counter
from typing import List, Optional, Tuple

from app.core.models import ProcessedData
from app.core.database import session_scope

class ProcessedDataService:
    def get_all_condition_names(self, processor_version: Optional[str] = None) -> List[str]:
        """
        Retrieve distinct condition names from the processed data stored in the database.

        Args:
            processor_version: Optional processor version filter (e.g., "v5"). If provided,
                only condition names produced by that processor version will be returned.

        Returns:
            A sorted list of unique condition names.

        Raises:
            RuntimeError: If the query fails.
        """
        try:
            with session_scope() as db:
                query = db.query(ProcessedData.condition_name)

                if processor_version:
                    query = query.filter(ProcessedData.processor_version == processor_version)

                return [name for (name,) in query.distinct().order_by(ProcessedData.condition_name.asc()).all()]
        except Exception as exc:
            raise RuntimeError(f"Failed to fetch condition names: {exc}") from exc

    def get_top_symptoms_across_conditions(
        self,
        top_n: int = 10,
        processor_version: Optional[str] = None,
    ) -> List[Tuple[str, int]]:
        """
        Find the most common symptoms across processed conditions stored in the database.

        Args:
            top_n: Number of top symptoms to return (default: 10)
            processor_version: Optional processor version filter (e.g., "v5")

        Returns:
            A list of tuples (symptom, count) sorted by count descending, limited to ``top_n``.

        Raises:
            RuntimeError: If database access fails.
        """
        try:
            with session_scope() as db:
                query = db.query(ProcessedData.condition_name, ProcessedData.content)

                if processor_version:
                    query = query.filter(ProcessedData.processor_version == processor_version)

                records = query.all()

            symptom_counts: Counter[str] = Counter()

            for condition_name, content in records:
                if not isinstance(content, dict):
                    continue

                symptoms_data = content.get("symptoms")

                if not (isinstance(symptoms_data, list) and len(symptoms_data) >= 2):
                    continue

                symptom_list = symptoms_data[1]

                if not isinstance(symptom_list, list):
                    continue

                unique_symptoms = {
                    symptom.strip()
                    for symptom in symptom_list
                    if isinstance(symptom, str) and symptom.strip()
                }

                for symptom in unique_symptoms:
                    symptom_counts[symptom] += 1

            return symptom_counts.most_common(top_n)
        except Exception as exc:
            raise RuntimeError(f"Failed to compute top symptoms: {exc}") from exc