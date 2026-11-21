import os

import pytest

# Importing the app builds the database engine and the LLM clients. Neither connects
# until used, so the tests run without a database or real API keys.
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:1/test")
os.environ.setdefault("GEMINI_API_KEY", "test")


@pytest.fixture
def anyio_backend():
    return "asyncio"


class InMemoryChatHistoryRepository:
    """Stands in for ChatHistoryRepository: the same stored format, kept in a dict."""

    def __init__(self):
        self.histories = {}

    def load(self, session_id):
        return self.histories.get(session_id, [])

    def save(self, session_id, history):
        self.histories[session_id] = history

    def clear(self, session_id):
        self.histories.pop(session_id, None)


class StubRetrievalService:
    """Stands in for RetrievalService: the same passages for every query."""

    def __init__(self, passages):
        self.passages = passages

    def search(self, query):
        return self.passages


@pytest.fixture
def chat_histories():
    return InMemoryChatHistoryRepository()
