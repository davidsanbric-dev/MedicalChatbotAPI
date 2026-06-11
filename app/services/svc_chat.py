import json
import logging

from langchain_core.runnables.history import RunnableWithMessageHistory
from app.schemas import RetrievalRequest
from app.services.svc_document import DocumentService
from app.core.memory import chat_message_history
from app.core.llm import llm
from app.core.prompts import query_rephraser_prompt_template, system_prompt_template

logger = logging.getLogger(__name__)


def _extract_text(content) -> str:
    """Coerce an LLM response `content` into a plain string.

    Some providers return a JSON string or a structured object instead of plain
    text; pull out a likely text field (`response`/`content`/`detail`) when so,
    otherwise stringify.
    """
    def _from_dict(d: dict) -> str:
        return d.get('response') or d.get('content') or d.get('detail') or str(d)

    if isinstance(content, str):
        try:
            parsed = json.loads(content)
        except Exception:
            return content
        return _from_dict(parsed) if isinstance(parsed, dict) else str(parsed)

    if isinstance(content, dict):
        return _from_dict(content)

    for attr in ('dict', 'to_dict'):
        method = getattr(content, attr, None)
        if callable(method):
            try:
                return _from_dict(method())
            except Exception:
                pass

    return str(content)


class ChatService:
    def __init__(self):
        self.document_service = DocumentService()

    async def get_conversational_response(self, payload: RetrievalRequest) -> str:
        """Generate a conversational response based on input and context."""

        conversational_chain = system_prompt_template | llm

        chain_with_history = RunnableWithMessageHistory(
            conversational_chain,
            chat_message_history,
            input_messages_key="input",
            history_messages_key="conversation_history",
        )
        
        conversation_history = chat_message_history(payload.session_id)

        rephrased_query = await conversation_history.rephrase_query_with_conversation(payload.query)
        
        retrieved_documents = self.document_service.get_similar_documents(
            RetrievalRequest(
                query=rephrased_query,
                session_id=payload.session_id,
            )
        )
        
        top_documents = self.document_service.filter_top_documents(retrieved_documents, rephrased_query, top_n=3)
        
        # If no relevant chunks found, return a fallback message
        if not top_documents:
            logger.warning("No relevant documents found for the query")
            config = {"configurable": {"session_id": payload.session_id}}
            
            fallback_response = await chain_with_history.ainvoke({
                "input": payload.query,
                "context": "No relevant information found in the knowledge base for this query.",
            }, config=config)

            return _extract_text(fallback_response.content)

        context_text = "".join(chunk['content'] for chunk in top_documents)

        config = {"configurable": {"session_id": payload.session_id}}
        
        generated_response = await chain_with_history.ainvoke({
            "input": payload.query,
            "context": context_text,
        }, config=config)

        return _extract_text(generated_response.content)
