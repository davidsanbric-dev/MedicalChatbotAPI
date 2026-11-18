"""chat_history: one row per chat session, holding its LangChain messages.

PostgresChatHistory is the same storage in the shape LangChain expects for conversation memory.
"""

from collections.abc import Sequence
from typing import Any

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict
from sqlalchemy import delete

from app.core.database import SessionLocal
from app.core.exceptions import ChatHistoryError
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


class PostgresChatHistory(BaseChatMessageHistory):
    def __init__(self, session_id: str):
        self.session_id = session_id

    @property
    def messages(self) -> list[BaseMessage]:
        try:
            return messages_from_dict(load(self.session_id))
        except Exception as error:
            raise ChatHistoryError("Couldn't load this conversation. Please start a new chat.") from error

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        history = messages_to_dict([*self.messages, *messages])
        try:
            save(self.session_id, history)
        except Exception as error:
            raise ChatHistoryError("Couldn't save this conversation. Please try again.") from error

    def clear(self) -> None:
        clear(self.session_id)
