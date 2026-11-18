"""The API's wire contract: request and response bodies, and the Server-Sent Events of the
two streams. The web frontend validates the same shapes with zod
(medical_conditions_react_frontend/src/lib/api.ts), so a change here belongs there too.

Bodies are Pydantic models, so FastAPI validates them and lists them in /docs. Stream
events are TypedDicts: the services yield plain dicts and the stream serializes them as they are.
"""

from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, StringConstraints

from app.domain.handoff import Urgency

# --- Requests ---


class ChatRequest(BaseModel):
    """One question in a conversation. `session_id` keys the conversation's memory."""

    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]
    session_id: Annotated[str, StringConstraints(min_length=1, max_length=100)]


# --- Shared parts ---


class HandoffTicket(TypedDict):
    """A handoff the agent opened: `id` is the request number shown to the visitor."""

    id: str
    urgency: Urgency


class ToolUse(TypedDict):
    name: str
    args: dict[str, Any]


# --- Responses ---


class ChatResponse(BaseModel):
    response: str


class AgentResponse(BaseModel):
    response: str
    sources: list[str]  # the conditions the answer drew on
    tools: list[ToolUse]  # in the order the agent called them
    handoff: HandoffTicket | None = None


class ErrorResponse(BaseModel):
    """Every error the API sends itself (429, 500). Validation errors (422) keep FastAPI's format."""

    detail: str


# --- Stream events: one `data: <json>` line each ---


class StageEvent(TypedDict):
    stage: Literal["rephrasing", "retrieving"]


class FoundEvent(TypedDict):
    stage: Literal["found"]
    count: int


class TokenEvent(TypedDict):
    token: str


class AnswerEvent(TypedDict):
    answer: str


class ToolEvent(TypedDict):
    tool: str
    args: dict[str, Any]


class HandoffEvent(TypedDict):
    handoff: HandoffTicket


class AgentAnswerEvent(TypedDict):
    answer: str
    sources: list[str]


class ErrorEvent(TypedDict):
    """The last event when a stream fails after it started (the status code is already sent)."""

    error: str


ChatEvent = StageEvent | FoundEvent | TokenEvent | AnswerEvent
AgentEvent = ToolEvent | HandoffEvent | AgentAnswerEvent
