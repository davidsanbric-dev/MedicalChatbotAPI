from langchain_core.runnables.history import RunnableWithMessageHistory
import json
from app.schemas import RetrievalRequest
from app.services.svc_document import DocumentService
from app.core.memory import chat_message_history
from app.core.llm import llm
from app.core.prompts import query_rephraser_prompt_template, system_prompt_template

class ChatService:
    def __init__(self):
        self.document_service = DocumentService()

    async def rephrase_query_with_conversation(self, query: str) -> str:
        """Rewrite the query based on conversation history."""

        conversation_aware_chain = query_rephraser_prompt_template | llm

        response = conversation_aware_chain.invoke({
            "conversation_history": self.messages,
            "input": query
        })

        print(f"Original query: '{query}' -> Rewritten query: '{response.content}'")

        return response.content
    
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
            print("[Warning] No relevant documents found for the query")
            config = {"configurable": {"session_id": payload.session_id}}
            
            fallback_response = await chain_with_history.ainvoke({
                "input": payload.query,
                "context": "No relevant information found in the knowledge base for this query.",
            }, config=config)

            # Ensure we return a plain string. If the LLM chain returns structured data,
            # extract likely text fields or fall back to stringifying.
            content = fallback_response.content
            print("[svc_chat] fallback_response.content type:", type(content), "repr:", repr(content))

            # If the content is a JSON string, try to parse it
            if isinstance(content, str):
                try:
                    parsed = json.loads(content)
                except Exception:
                    return content
                else:
                    if isinstance(parsed, dict):
                        return parsed.get('response') or parsed.get('content') or parsed.get('detail') or str(parsed)
                    return str(parsed)

            # If it's a mapping-like object
            if isinstance(content, dict):
                return content.get('response') or content.get('content') or content.get('detail') or str(content)

            # If it's an object with dict()/to_dict(), try to convert
            if hasattr(content, 'dict') and callable(getattr(content, 'dict')):
                try:
                    d = content.dict()
                    return d.get('response') or d.get('content') or d.get('detail') or str(d)
                except Exception:
                    pass

            if hasattr(content, 'to_dict') and callable(getattr(content, 'to_dict')):
                try:
                    d = content.to_dict()
                    return d.get('response') or d.get('content') or d.get('detail') or str(d)
                except Exception:
                    pass

            # Fallback
            return str(content)

        context_text = "".join(chunk['content'] for chunk in top_documents)

        config = {"configurable": {"session_id": payload.session_id}}
        
        generated_response = await chain_with_history.ainvoke({
            "input": payload.query,
            "context": context_text,
        }, config=config)

        content = generated_response.content
        print("[svc_chat] generated_response.content type:", type(content), "repr:", repr(content))

        if isinstance(content, str):
            # If the string is JSON, try to extract inner fields
            try:
                parsed = json.loads(content)
            except Exception:
                return content
            else:
                if isinstance(parsed, dict):
                    return parsed.get('response') or parsed.get('content') or parsed.get('detail') or str(parsed)
                return str(parsed)

        if isinstance(content, dict):
            return content.get('response') or content.get('content') or content.get('detail') or str(content)

        if hasattr(content, 'dict') and callable(getattr(content, 'dict')):
            try:
                d = content.dict()
                return d.get('response') or d.get('content') or d.get('detail') or str(d)
            except Exception:
                pass

        if hasattr(content, 'to_dict') and callable(getattr(content, 'to_dict')):
            try:
                d = content.to_dict()
                return d.get('response') or d.get('content') or d.get('detail') or str(d)
            except Exception:
                pass

        return str(content)
