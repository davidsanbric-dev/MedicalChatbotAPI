"""handoff_requests: conversations the agent escalated to a human."""

import uuid

from sqlalchemy import delete

from app.core.database import SessionLocal
from app.core.models import HandoffRequest


def create(session_id: str, urgency: str, reason: str, last_message: str) -> uuid.UUID:
    """Open a request and return its id (the ticket number shown to the visitor)."""
    with SessionLocal.begin() as session:
        request = HandoffRequest(session_id=session_id, urgency=urgency, reason=reason, last_message=last_message)
        session.add(request)
        session.flush()
        return request.id


def delete_for_sessions(prefix: str) -> int:
    """Remove the requests of every session starting with `prefix` (the agent evaluation's)."""
    with SessionLocal.begin() as session:
        result = session.execute(delete(HandoffRequest).where(HandoffRequest.session_id.startswith(prefix)))
        return result.rowcount
