from typing import Annotated

from pydantic import BaseModel, StringConstraints


class ChatRequest(BaseModel):
    """One question in a conversation. `session_id` keys the conversation's memory."""

    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]
    session_id: Annotated[str, StringConstraints(min_length=1, max_length=100)]
