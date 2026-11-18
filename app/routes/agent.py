"""Agent endpoints: the tool-calling assistant (app/services/agent.py), rate limited like chat."""

from fastapi import APIRouter

from app.routes.streaming import EventStream, sse_response
from app.schemas import AgentResponse, ChatRequest
from app.services import agent

router = APIRouter()


@router.post("/")
async def run(request: ChatRequest) -> AgentResponse:
    """The answer, the conditions it drew on, the tools called and the handoff opened (or null)."""
    return await agent.run(request)


@router.post("/stream", response_class=EventStream)
async def stream_run(request: ChatRequest) -> EventStream:
    """Server-Sent Events: one `data: <json>` line per AgentEvent (app/schemas.py)."""
    return sse_response(agent.stream_run(request), request.session_id)
