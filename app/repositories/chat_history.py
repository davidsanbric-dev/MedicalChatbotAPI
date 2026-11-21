"""chat_history: one row per chat session, holding its LangChain messages.

PostgresChatHistory is the same storage in the shape LangChain expects for conversation memory.
"""

from collections.abc import Sequence
from typing import Any

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict

from app.core.exceptions import ChatHistoryError
from app.core.models import ChatHistory
from app.repositories.base import BaseRepository


class ChatHistoryRepository(BaseRepository):
    model = ChatHistory

    def load(self, session_id: str) -> list[dict[str, Any]]:
        """The session's messages as stored (LangChain's dict format); empty for a new session."""
        with self._session() as session:
            row = session.get(ChatHistory, session_id)
            return row.history if row else []

    def save(self, session_id: str, history: list[dict[str, Any]]) -> None:
        """Insert the session's row, or overwrite its history."""
        with self._transaction() as session:
            session.merge(ChatHistory(session_id=session_id, history=history))

    def clear(self, session_id: str) -> None:
        self._delete_where(ChatHistory.session_id == session_id)


class PostgresChatHistory(BaseChatMessageHistory):
    """One session's conversation, read and written through ChatHistoryRepository.

    Failures become ChatHistoryError, whose message is safe to show the visitor.
    """

    def __init__(self, session_id: str, repository: ChatHistoryRepository):
        self.session_id = session_id
        self.repository = repository

    @property
    def messages(self) -> list[BaseMessage]:
        try:
            return messages_from_dict(self.repository.load(self.session_id))
        except Exception as error:
            raise ChatHistoryError("Couldn't load this conversation. Please start a new chat.") from error

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        history = messages_to_dict([*self.messages, *messages])
        try:
            self.repository.save(self.session_id, history)
        except Exception as error:
            raise ChatHistoryError("Couldn't save this conversation. Please try again.") from error

    def clear(self) -> None:
        self.repository.clear(self.session_id)
