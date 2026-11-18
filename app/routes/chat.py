"""Chat endpoints, rate limited in app/main.py (app/routes/rate_limit.py). All endpoints:
https://medicalchatbot.demos.davieldev.com/api#endpoints
"""

from fastapi import APIRouter

from app.routes.streaming import EventStream, sse_response
from app.schemas import ChatRequest, ChatResponse
from app.services import chat

router = APIRouter()


@router.post("/")
async def reply(request: ChatRequest) -> ChatResponse:
    return ChatResponse(response=await chat.reply(request))


@router.post("/stream", response_class=EventStream)
async def stream_reply(request: ChatRequest) -> EventStream:
    """Server-Sent Events: one `data: <json>` line per ChatEvent (app/schemas.py)."""
    return sse_response(chat.stream_reply(request), request.session_id)
