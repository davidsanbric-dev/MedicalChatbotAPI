"""Agent endpoints: the tool-calling assistant (app/services/agent.py), rate limited like chat."""

from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.routes.streaming import sse_response
from app.schemas import ChatRequest
from app.services import agent

router = APIRouter()


@router.post("/")
async def run(request: ChatRequest) -> dict[str, Any]:
    """The answer, the conditions it drew on, the tools called and the handoff opened (or null)."""
    return await agent.run(request)


@router.post("/stream")
async def stream_run(request: ChatRequest) -> StreamingResponse:
    """Server-Sent Events: one `data: <json>` line per event of agent.stream_run."""
    return sse_response(agent.stream_run(request), request.session_id)
