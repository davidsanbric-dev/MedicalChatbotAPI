"""Agent endpoints: the tool-calling assistant (app/services/agent.py), rate limited like chat."""

from fastapi import APIRouter

from app.dependencies import AgentServiceDep
from app.routes.streaming import EventStream, sse_response
from app.schemas import AgentResponse, ChatRequest

router = APIRouter()


@router.post("/")
async def run(request: ChatRequest, agent_service: AgentServiceDep) -> AgentResponse:
    """The answer, the conditions it drew on, the tools called and the handoff opened (or null)."""
    return await agent_service.reply(request)


@router.post("/stream", response_class=EventStream)
async def stream_run(request: ChatRequest, agent_service: AgentServiceDep) -> EventStream:
    """Server-Sent Events: one `data: <json>` line per AgentEvent (app/schemas.py)."""
    return sse_response(agent_service.stream(request), request.session_id)
