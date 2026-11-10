"""Conversation memory for LangChain, kept in the chat_history table (one row per session)."""

from collections.abc import Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict

from app.core.exceptions import ChatHistoryError
from app.repositories import chat_history


class PostgresChatHistory(BaseChatMessageHistory):
    def __init__(self, session_id: str):
        self.session_id = session_id

    @property
    def messages(self) -> list[BaseMessage]:
        try:
            return messages_from_dict(chat_history.load(self.session_id))
        except Exception as error:
            raise ChatHistoryError("Couldn't load this conversation. Please start a new chat.") from error

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        history = messages_to_dict([*self.messages, *messages])
        try:
            chat_history.save(self.session_id, history)
        except Exception as error:
            raise ChatHistoryError("Couldn't save this conversation. Please try again.") from error

    def clear(self) -> None:
        chat_history.clear(self.session_id)
