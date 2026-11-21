"""handoff_requests: conversations the agent escalated to a human."""

from app.core.models import HandoffRequest
from app.domain.handoff import Handoff
from app.repositories.base import BaseRepository


class HandoffRepository(BaseRepository):
    model = HandoffRequest

    def add(self, handoff: Handoff) -> None:
        """Open the request; its id is already the ticket number shown to the visitor."""
        with self._transaction() as session:
            session.add(self._row_for(handoff))

    def delete_by_session_prefix(self, prefix: str) -> int:
        """Remove the requests of every session starting with `prefix` (the agent evaluation's)."""
        return self._delete_where(HandoffRequest.session_id.startswith(prefix))

    @staticmethod
    def _row_for(handoff: Handoff) -> HandoffRequest:
        return HandoffRequest(
            id=handoff.id,
            session_id=handoff.session_id,
            urgency=handoff.urgency.value,
            reason=handoff.reason,
            last_message=handoff.last_message,
        )
