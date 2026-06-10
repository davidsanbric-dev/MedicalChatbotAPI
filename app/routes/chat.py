from fastapi import APIRouter, HTTPException

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