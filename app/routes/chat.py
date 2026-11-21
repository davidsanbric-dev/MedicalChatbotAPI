"""Chat endpoints, rate limited in app/main.py (app/routes/rate_limit.py). All endpoints:
https://medicalchatbot.demos.davieldev.com/api#endpoints
"""

from fastapi import APIRouter

from app.dependencies import ChatServiceDep
from app.routes.streaming import EventStream, sse_response
from app.schemas import ChatRequest, ChatResponse

router = APIRouter()


@router.post("/")
async def reply(request: ChatRequest, chat_service: ChatServiceDep) -> ChatResponse:
    return await chat_service.reply(request)


@router.post("/stream", response_class=EventStream)
async def stream_reply(request: ChatRequest, chat_service: ChatServiceDep) -> EventStream:
    """Server-Sent Events: one `data: <json>` line per ChatEvent (app/schemas.py)."""
    return sse_response(chat_service.stream(request), request.session_id)
