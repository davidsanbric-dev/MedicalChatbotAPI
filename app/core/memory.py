from typing import List

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict

from app.core.exceptions import ChatHistoryError
from app.repositories.rps_chat import ChatHistoryRepository


class PostgreSQLChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.repo = ChatHistoryRepository()

    @property
    def messages(self) -> List[BaseMessage]:
        """Retrieve chat history for a session from PostgreSQL."""
        try:
            history = self.repo.find_by_session(self.session_id)
            if not history:
                return []
            return messages_from_dict(history.history or [])
        except Exception as e:
            raise ChatHistoryError(f"Failed to read chat history for session {self.session_id}") from e

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """Append messages to the session's history in PostgreSQL."""
        try:
            all_messages = self.messages + messages
            self.repo.save(self.session_id, messages_to_dict(all_messages))
        except Exception as e:
            raise ChatHistoryError(f"Failed to save chat history for session {self.session_id}") from e

    def clear(self) -> None:
        """Delete the session's history from PostgreSQL."""
        try:
            self.repo.delete_by_session(self.session_id)
        except Exception as e:
            raise ChatHistoryError(f"Failed to delete chat history for session {self.session_id}") from e


# Callable to get chat history instance
def chat_message_history(session_id: str) -> PostgreSQLChatMessageHistory:
    return PostgreSQLChatMessageHistory(session_id=session_id)