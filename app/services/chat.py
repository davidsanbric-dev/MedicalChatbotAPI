"""One chat turn: understand the question, find passages, stream the answer.

These are the four steps on https://medicalchatbot.demos.davieldev.com/api. The events
(ChatEvent in app/schemas.py) are what the chat shows while it waits
(https://medicalchatbot.demos.davieldev.com/interface#streaming):
    {"stage": "rephrasing"}             understanding the question
    {"stage": "retrieving"}             searching the knowledge base
    {"stage": "found", "count": int}    passages selected as context
    {"token": str}                      the next piece of the answer
    {"answer": str}                     the final answer
"""

import asyncio
import logging
from collections.abc import AsyncIterator

from langchain_core.runnables.history import RunnableWithMessageHistory

from app.core.llm import llm, model_name_of, text_of, unwrap_json_answer
from app.core.prompts import ANSWER_PROMPT, REPHRASE_PROMPT, answer_context
from app.repositories.chat_history import PostgresChatHistory
from app.schemas import ChatEvent, ChatRequest
from app.services import retrieval

logger = logging.getLogger(__name__)

_rephrase_chain = REPHRASE_PROMPT | llm
# Loads the session's history into the prompt, and saves the new turn once answered.
_answer_chain = RunnableWithMessageHistory(
    ANSWER_PROMPT | llm,
    PostgresChatHistory,
    input_messages_key="input",
    history_messages_key="conversation_history",
)


async def stream_reply(request: ChatRequest) -> AsyncIterator[ChatEvent]:
    yield {"stage": "rephrasing"}
    question = await _rephrase(request)

    yield {"stage": "retrieving"}
    # Embedding, SQL and re-ranking block; in a worker thread they can't stall other visitors' streams.
    passages = await asyncio.to_thread(retrieval.search, question)
    if not passages:
        logger.warning("No relevant passages for %r", question)
    yield {"stage": "found", "count": len(passages)}

    async for event in _stream_answer(request, answer_context(passages)):
        yield event


async def reply(request: ChatRequest) -> str:
    """The non-streaming variant: only the final answer."""
    answer = ""
    async for event in stream_reply(request):
        answer = event.get("answer", answer)
    return answer


async def _rephrase(request: ChatRequest) -> str:
    """The question as a standalone search query, in English like the knowledge base."""
    history = await PostgresChatHistory(request.session_id).aget_messages()
    response = await _rephrase_chain.ainvoke({"conversation_history": history, "input": request.query})
    # Models sometimes wrap the query in quotes, which would pollute the embedding.
    question = text_of(response.content).strip().strip("\"'").strip()
    logger.info("Rephrased [model=%s]: %r -> %r", model_name_of(response), request.query, question)
    return question


async def _stream_answer(request: ChatRequest, context: str) -> AsyncIterator[ChatEvent]:
    parts: list[str] = []
    model = "unknown"
    config = {"configurable": {"session_id": request.session_id}}
    async for chunk in _answer_chain.astream({"input": request.query, "context": context}, config=config):
        if (name := model_name_of(chunk)) != "unknown":
            model = name
        if token := text_of(chunk.content):
            parts.append(token)
            yield {"token": token}
    logger.info("Answer generated [model=%s]", model)
    yield {"answer": unwrap_json_answer("".join(parts))}
