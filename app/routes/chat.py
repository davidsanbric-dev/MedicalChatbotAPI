import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.services.svc_document import DocumentService
from app.services.svc_chat import ChatService
from app.schemas import RetrievalRequest
from app.core.exceptions import RetrievalError

document_service = DocumentService()
chat_service = ChatService()

router = APIRouter()

@router.post('/')
async def reply_with_rag_and_memory(payload: RetrievalRequest):
    try:
        conversational_response = await chat_service.get_conversational_response(payload)
        return {'response': conversational_response}
    except RetrievalError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/stream')
async def reply_with_rag_and_memory_stream(payload: RetrievalRequest):
    """Server-Sent Events variant: emits progress stages then the final answer.

    Each line is `data: <json>\\n\\n`, where json is one of the events produced
    by ChatService.stream_conversational_response (or {"error": ...} on failure).
    """
    async def event_generator():
        try:
            async for event in chat_service.stream_conversational_response(payload):
                yield f"data: {json.dumps(event)}\n\n"
        except RetrievalError as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )