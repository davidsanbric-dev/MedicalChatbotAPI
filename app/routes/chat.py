import json
import logging

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.services.svc_chat import ChatService
from app.schemas import RetrievalRequest

logger = logging.getLogger(__name__)

chat_service = ChatService()

router = APIRouter()

@router.post('/')
async def reply_with_rag_and_memory(payload: RetrievalRequest):
    # Service errors (e.g. RetrievalError) are mapped to HTTP by the app-level
    # exception handlers in app/main.py.
    conversational_response = await chat_service.get_conversational_response(payload)
    return {'response': conversational_response}


@router.post('/stream')
async def reply_with_rag_and_memory_stream(payload: RetrievalRequest):
    """Server-Sent Events variant: emits progress stages then the final answer.

    Each line is `data: <json>\\n\\n`, where json is one of the events produced
    by ChatService.stream_conversational_response (or {"error": ...} on failure).
    """
    async def event_generator():
        # Errors must be surfaced as an SSE event: once streaming starts the HTTP
        # status is already sent, so the app-level exception handlers can't run.
        try:
            async for event in chat_service.stream_conversational_response(payload):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            logger.exception("Chat stream failed for session %s", payload.session_id)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )