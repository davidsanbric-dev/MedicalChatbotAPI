import os

import pytest

# Importing the app builds the database engine and the LLM clients. Neither connects
# until used, so the tests run without a database or real API keys.
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:1/test")
os.environ.setdefault("GEMINI_API_KEY", "test")


@pytest.fixture
def anyio_backend():
    return "asyncio"
