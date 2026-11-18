"""handoff_requests: conversations the agent escalated to a human."""

from sqlalchemy import delete

from app.core.database import SessionLocal
from app.core.models import HandoffRequest
from app.domain.handoff import Handoff


def add(handoff: Handoff) -> None:
    """Open the request; its id is already the ticket number shown to the visitor."""
    with SessionLocal.begin() as session:
        session.add(
            HandoffRequest(
                id=handoff.id,
                session_id=handoff.session_id,
                urgency=handoff.urgency.value,
                reason=handoff.reason,
                last_message=handoff.last_message,
            )
        )


def delete_for_sessions(prefix: str) -> int:
    """Remove the requests of every session starting with `prefix` (the agent evaluation's)."""
    with SessionLocal.begin() as session:
        result = session.execute(delete(HandoffRequest).where(HandoffRequest.session_id.startswith(prefix)))
        return result.rowcount
