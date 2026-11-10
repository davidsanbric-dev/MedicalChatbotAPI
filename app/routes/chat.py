"""Chat endpoints, rate limited in app/main.py. All endpoints:
https://medicalchatbot.demos.davieldev.com/api#endpoints
"""

import json
import logging

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.core.exceptions import public_message
from app.schemas import ChatRequest
from app.services import chat

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/")
async def reply(request: ChatRequest) -> dict[str, str]:
    return {"response": await chat.reply(request)}


@router.post("/stream")
async def stream_reply(request: ChatRequest) -> StreamingResponse:
    """Server-Sent Events: one `data: <json>` line per event of chat.stream_reply."""

    async def events():
        # Once streaming starts the status code is already sent, so a failure goes out
        # as a final event instead of through the app's exception handler.
        try:
            async for event in chat.stream_reply(request):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as error:
            logger.exception("Chat stream failed for session %s", request.session_id)
            yield f"data: {json.dumps({'error': public_message(error)})}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
