"""What the two conversational services share: ChatService (app/services/chat.py) and
AgentService (app/services/agent.py) each answer one visitor message within a session.

Both answer as a stream of events (the /stream endpoints). The plain endpoints get the same
stream collected into one response, so the steps of a turn are written only once.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from app.repositories.chat_history import ChatHistoryRepository, PostgresChatHistory
from app.schemas import ChatRequest


class ConversationService[EventT, ResponseT](ABC):
    def __init__(self, chat_history_repository: ChatHistoryRepository) -> None:
        self.chat_history_repository = chat_history_repository

    def chat_history(self, session_id: str) -> PostgresChatHistory:
        """The session's conversation so far, in the shape LangChain reads and writes."""
        return PostgresChatHistory(session_id, self.chat_history_repository)

    @abstractmethod
    def stream(self, request: ChatRequest) -> AsyncIterator[EventT]:
        """Answer the message one event at a time, as the turn happens."""

    async def reply(self, request: ChatRequest) -> ResponseT:
        """Answer the message in one response: the events of `stream`, collected."""
        response = self._new_response()
        async for event in self.stream(request):
            self._add_to_response(response, event)
        return response

    @abstractmethod
    def _new_response(self) -> ResponseT:
        """The response before any event has arrived."""

    @abstractmethod
    def _add_to_response(self, response: ResponseT, event: EventT) -> None:
        """Copy what one event tells into the response."""
