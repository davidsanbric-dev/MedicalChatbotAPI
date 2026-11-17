"""Chat endpoints, rate limited in app/main.py. All endpoints:
https://medicalchatbot.demos.davieldev.com/api#endpoints
"""

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.routes.streaming import sse_response
from app.schemas import ChatRequest
from app.services import chat

router = APIRouter()


@router.post("/")
async def reply(request: ChatRequest) -> dict[str, str]:
    return {"response": await chat.reply(request)}


@router.post("/stream")
async def stream_reply(request: ChatRequest) -> StreamingResponse:
    """Server-Sent Events: one `data: <json>` line per event of chat.stream_reply."""
    return sse_response(chat.stream_reply(request), request.session_id)
