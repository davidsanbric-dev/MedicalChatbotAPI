"""chat_history: one row per chat session, holding its LangChain messages."""

from typing import Any

from sqlalchemy import delete

from app.core.database import SessionLocal
from app.core.models import ChatHistory


def load(session_id: str) -> list[dict[str, Any]]:
    with SessionLocal() as session:
        row = session.get(ChatHistory, session_id)
        return row.history if row else []


def save(session_id: str, history: list[dict[str, Any]]) -> None:
    """Insert the session's row, or overwrite its history."""
    with SessionLocal.begin() as session:
        session.merge(ChatHistory(session_id=session_id, history=history))


def clear(session_id: str) -> None:
    with SessionLocal.begin() as session:
        session.execute(delete(ChatHistory).where(ChatHistory.session_id == session_id))
