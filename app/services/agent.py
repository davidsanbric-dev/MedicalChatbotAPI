"""The tool-calling agent: the model decides, turn by turn, whether to search the knowledge
base, list what it covers, or hand the conversation to a human.

Where the RAG chat (app/services/chat.py) always runs the same fixed steps, here the model
chooses the steps. One turn is a loop:

    1. Send the conversation and the tool descriptions to the model.
    2. If it asks for tools, run them and send back their results (tool messages).
    3. Repeat until it answers in plain text, at most MAX_STEPS model calls.

Guardrails around the loop:
    - Emergency wording (app/domain/triage.py) is caught before the model is asked: it opens
      an "emergency" handoff and returns fixed safety advice, so a model mistake can't miss it.
    - Tool arguments are validated by each tool's schema; a failing tool returns an error
      message to the model instead of ending the turn.
    - MAX_STEPS bounds the cost and latency of a model that keeps calling tools.

Events (AgentEvent in app/schemas.py), in the style of the chat stream:
    {"tool": str, "args": dict}                   a tool the agent is calling
    {"handoff": {"id": str, "urgency": str}}      a human handoff was opened
    {"answer": str, "sources": [str]}             the final answer and the conditions it used
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from app.core.llm import model_name_of, text_of, unwrap_json_answer, with_tools
from app.core.prompts import AGENT_SYSTEM_PROMPT, search_results
from app.domain.handoff import Handoff, Urgency
from app.domain.triage import EMERGENCY_ADVICE, is_emergency
from app.repositories import handoffs
from app.repositories.chat_history import PostgresChatHistory
from app.schemas import AgentEvent, AgentResponse, ChatRequest, HandoffEvent
from app.services import insights, retrieval

logger = logging.getLogger(__name__)

MAX_STEPS = 4  # model calls per turn

STEP_LIMIT_ANSWER = (
    "I couldn't finish answering that. Please rephrase your question, or ask me to connect you with a person."
)


class SearchArgs(BaseModel):
    query: str = Field(min_length=2, max_length=300, description="A short search query in English.")


class HandoffArgs(BaseModel):
    reason: str = Field(min_length=3, max_length=500, description="Why a human is needed, in one sentence.")
    urgency: Urgency = Field(
        description='"emergency" for life-threatening signs, "urgent" for care needed today, else "routine".'
    )


@dataclass
class Turn:
    """What the tools learn during one turn, read back by the loop to build its events."""

    session_id: str
    message: str
    answer: str = ""
    sources: list[str] = field(default_factory=list)
    handoff: Handoff | None = None


def build_tools(turn: Turn) -> list[BaseTool]:
    """The agent's tools for one turn; they record their side effects on `turn`."""

    def search_knowledge_base(query: str) -> str:
        passages = retrieval.search(query)
        for passage in passages:
            if passage.condition and passage.condition not in turn.sources:
                turn.sources.append(passage.condition)
        return search_results(passages)

    def list_conditions() -> str:
        return "The knowledge base covers: " + ", ".join(insights.condition_names())

    def request_human_handoff(reason: str, urgency: Urgency) -> str:
        handoff = Handoff(turn.session_id, Urgency(urgency), reason, turn.message)
        handoffs.add(handoff)
        turn.handoff = handoff
        return f"Handoff request {handoff.id} opened with urgency {handoff.urgency}. A person from the care team will follow up."

    return [
        StructuredTool.from_function(
            search_knowledge_base,
            description="Search the medical knowledge base for passages about symptoms, causes and treatments.",
            args_schema=SearchArgs,
        ),
        StructuredTool.from_function(
            list_conditions, description="List the conditions the knowledge base covers."
        ),
        StructuredTool.from_function(
            request_human_handoff,
            description="Hand the conversation to a human from the care team.",
            args_schema=HandoffArgs,
        ),
    ]


async def stream_run(request: ChatRequest) -> AsyncIterator[AgentEvent]:
    turn = Turn(request.session_id, request.query)
    memory = PostgresChatHistory(request.session_id)

    if is_emergency(request.query):
        async for event in _escalate_emergency(turn):
            yield event
    else:
        history = await memory.aget_messages()
        async for event in _loop(turn, history):
            yield event

    # Only the visitor's message and the final answer are kept: the next turn needs the
    # conversation, not this turn's tool traffic.
    await memory.aadd_messages([HumanMessage(request.query), AIMessage(turn.answer)])
    yield {"answer": turn.answer, "sources": turn.sources}


async def run(request: ChatRequest) -> AgentResponse:
    """The non-streaming variant: the answer, the tools called and the handoff, if any."""
    result = AgentResponse(response="", sources=[], tools=[])
    async for event in stream_run(request):
        if "tool" in event:
            result.tools.append({"name": event["tool"], "args": event["args"]})
        elif "handoff" in event:
            result.handoff = event["handoff"]
        elif "answer" in event:
            result.response, result.sources = event["answer"], event["sources"]
    return result


async def _escalate_emergency(turn: Turn) -> AsyncIterator[AgentEvent]:
    handoff = Handoff.for_emergency(turn.session_id, turn.message)
    yield {"tool": "request_human_handoff", "args": {"reason": handoff.reason, "urgency": handoff.urgency}}
    await asyncio.to_thread(handoffs.add, handoff)
    turn.handoff = handoff
    logger.warning("Emergency guardrail opened handoff %s for session %s", handoff.id, turn.session_id)
    yield _handoff_event(handoff)
    turn.answer = EMERGENCY_ADVICE.format(id=handoff.id)


def _handoff_event(handoff: Handoff) -> HandoffEvent:
    return {"handoff": {"id": str(handoff.id), "urgency": handoff.urgency}}


async def _loop(turn: Turn, history: list[BaseMessage]) -> AsyncIterator[AgentEvent]:
    tools = build_tools(turn)
    by_name = {tool.name: tool for tool in tools}
    model = with_tools(tools)
    messages: list[BaseMessage] = [SystemMessage(AGENT_SYSTEM_PROMPT), *history, HumanMessage(turn.message)]

    for step in range(1, MAX_STEPS + 1):
        response = await model.ainvoke(messages)
        messages.append(response)
        if not response.tool_calls:
            turn.answer = unwrap_json_answer(text_of(response.content).strip())
            logger.info("Agent answered after %d step(s) [model=%s]", step, model_name_of(response))
            return
        for call in response.tool_calls:
            yield {"tool": call["name"], "args": call["args"]}
            handoff_before = turn.handoff
            result = await _call_tool(by_name, call)
            messages.append(ToolMessage(result, tool_call_id=call["id"], name=call["name"]))
            if turn.handoff is not None and turn.handoff is not handoff_before:
                yield _handoff_event(turn.handoff)

    logger.warning("Agent hit the %d-step limit for session %s", MAX_STEPS, turn.session_id)
    turn.answer = STEP_LIMIT_ANSWER


async def _call_tool(by_name: dict[str, BaseTool], call: ToolCall) -> str:
    """Run one tool call; any failure goes back to the model as text it can react to."""
    tool = by_name.get(call["name"])
    if tool is None:
        return f"Error: there is no tool named {call['name']!r}."
    try:
        # Tools hit the database and the ONNX models: off the event loop, like the chat's retrieval.
        return await asyncio.to_thread(tool.invoke, call["args"])
    except Exception as error:
        logger.warning("Tool %s failed: %s", call["name"], error)
        return f"Error: the tool {call['name']} failed ({type(error).__name__}). Try different arguments or answer without it."
