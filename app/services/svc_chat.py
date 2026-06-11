import json
import logging

from langchain_core.runnables.history import RunnableWithMessageHistory
from app.schemas import RetrievalRequest
from app.services.svc_document import DocumentService
from app.core.memory import chat_message_history
from app.core.llm import llm, model_name_of
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

    async def _rephrase_query(self, conversation_history, query: str) -> str:
        """Rewrite `query` as a standalone search query, informed by chat history.

        The knowledge base is in English, so the rephraser also normalizes the
        query to English (see query_rephraser_prompt_template).
        """
        chain = query_rephraser_prompt_template | llm
        response = chain.invoke({
            "conversation_history": conversation_history.messages,
            "input": query,
        })

        # The model sometimes wraps the rephrased query in surrounding quotes,
        # which would otherwise pollute the embedding query text.
        rephrased = response.content.strip().strip('"\'').strip()

        logger.info(
            "Rephrased query [model=%s]: '%s' -> '%s'",
            model_name_of(response),
            query,
            rephrased,
        )

        return rephrased

    async def _stream_answer(self, chain_with_history, inputs, config):
        """Token-stream an answer from the chain.

        Yields {"token": str} for each delta, then a final {"answer": str} with
        the cleaned full text. Logs the model that actually produced the answer.
        """
        accumulated = ""
        model = "unknown"
        async for chunk in chain_with_history.astream(inputs, config=config):
            found = model_name_of(chunk)
            if found != "unknown":
                model = found

            delta = chunk.content
            if isinstance(delta, list):
                # Some providers return structured content parts; flatten to text.
                delta = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in delta
                )
            if delta:
                accumulated += delta
                yield {"token": delta}

        logger.info("Answer generated [model=%s]", model)
        # Clean the full text (handles rare JSON-wrapped responses) so the stored
        # final message is correct even when streamed deltas were raw.
        yield {"answer": _extract_text(accumulated)}

    async def stream_conversational_response(self, payload: RetrievalRequest):
        """Run the RAG pipeline, yielding progress events then the final answer.

        Yields dicts the route serializes as SSE:
        - {"stage": "rephrasing"}           understanding the question
        - {"stage": "retrieving"}           searching the knowledge base
        - {"stage": "found", "count": int}  number of passages selected as context
        - {"token": str}                    a streamed delta of the answer
        - {"answer": str}                   the final, cleaned assistant message

        This is the single source of truth for the chat pipeline;
        get_conversational_response() just drains it for the non-streaming route.
        """
        conversational_chain = system_prompt_template | llm

        chain_with_history = RunnableWithMessageHistory(
            conversational_chain,
            chat_message_history,
            input_messages_key="input",
            history_messages_key="conversation_history",
        )

        conversation_history = chat_message_history(payload.session_id)

        yield {"stage": "rephrasing"}
        rephrased_query = await self._rephrase_query(conversation_history, payload.query)

        yield {"stage": "retrieving"}
        retrieved_documents = self.document_service.get_similar_documents(
            RetrievalRequest(
                query=rephrased_query,
                session_id=payload.session_id,
            )
        )

        top_documents = self.document_service.filter_top_documents(retrieved_documents, rephrased_query, top_n=5)

        config = {"configurable": {"session_id": payload.session_id}}

        # If no relevant chunks found, fall back to a no-context answer.
        if not top_documents:
            logger.warning("No relevant documents found for the query")
            yield {"stage": "found", "count": 0}

            async for event in self._stream_answer(
                chain_with_history,
                {
                    "input": payload.query,
                    "context": "No relevant information found in the knowledge base for this query.",
                },
                config,
            ):
                yield event
            return

        yield {"stage": "found", "count": len(top_documents)}

        # Separate chunks so the model reads them as distinct passages rather
        # than one run-on blob.
        context_text = "\n\n".join(chunk['content'] for chunk in top_documents)

        async for event in self._stream_answer(
            chain_with_history,
            {"input": payload.query, "context": context_text},
            config,
        ):
            yield event

    async def get_conversational_response(self, payload: RetrievalRequest) -> str:
        """Generate a conversational response based on input and context.

        Non-streaming variant: drains the streaming pipeline and returns only
        the final answer.
        """
        answer = ""
        async for event in self.stream_conversational_response(payload):
            if "answer" in event:
                answer = event["answer"]
        return answer
