"""The tool-calling agent: the model decides, turn by turn, whether to search the knowledge
base, list what it covers, or hand the conversation to a human.

Where the RAG chat (app/services/chat.py) always runs the same fixed steps, here the model
chooses the steps. One turn is a loop:

    1. Send the conversation and the tool descriptions to the model.
    2. If it asks for tools, run them and send back their results (tool messages).
    3. Repeat until it answers in plain text, at most MAX_STEPS model calls.

Guardrails around the loop:
    - Emergency wording is caught before the model is asked: it opens an "emergency"
      handoff and returns fixed safety advice, so a model mistake can't miss it.
    - Tool arguments are validated by each tool's schema; a failing tool returns an error
      message to the model instead of ending the turn.
    - MAX_STEPS bounds the cost and latency of a model that keeps calling tools.

Events, in the style of the chat stream:
    {"tool": str, "args": dict}                   a tool the agent is calling
    {"handoff": {"id": str, "urgency": str}}      a human handoff was opened
    {"answer": str, "sources": [str]}             the final answer and the conditions it used
"""

import asyncio
import logging
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from app.core.llm import model_name_of, text_of, with_tools
from app.core.memory import PostgresChatHistory
from app.core.prompts import AGENT_SYSTEM_PROMPT
from app.repositories import handoffs
from app.schemas import ChatRequest
from app.services import insights, retrieval

logger = logging.getLogger(__name__)

MAX_STEPS = 4  # model calls per turn

STEP_LIMIT_ANSWER = (
    "I couldn't finish answering that. Please rephrase your question, or ask me to connect you with a person."
)
EMERGENCY_ANSWER = (
    "This may be a medical emergency. Please call your local emergency number now "
    "(123 in Colombia, 911 in the US and Mexico) or go to the nearest emergency room. "
    "I have also alerted our care team (request {id})."
)

# English and Spanish wording that must never wait for a model's judgement.
EMERGENCY_PATTERN = re.compile(
    r"chest pain|can'?t breathe|cannot breathe|trouble breathing|struggling to breathe|"
    r"suicid|kill(?:ing)? myself|end(?:ing)? my (?:own )?life|take my (?:own )?life|want to die|"
    r"overdos|unconscious|passed out|"
    r"face (?:is )?drooping|slurred speech|severe bleeding|bleeding (?:heavily|a lot)|"
    r"dolor (?:de|en el) pecho|no puedo respirar|me cuesta respirar|quitarme la vida|"
    r"suicidarme|matarme|sobredosis|inconsciente|sangrado (?:fuerte|abundante)",
    re.IGNORECASE,
)


class SearchArgs(BaseModel):
    query: str = Field(min_length=2, max_length=300, description="A short search query in English.")


class HandoffArgs(BaseModel):
    reason: str = Field(min_length=3, max_length=500, description="Why a human is needed, in one sentence.")
    urgency: Literal["routine", "urgent", "emergency"] = Field(
        description='"emergency" for life-threatening signs, "urgent" for care needed today, else "routine".'
    )


@dataclass
class Turn:
    """What the tools learn during one turn, read back by the loop to build its events."""

    session_id: str
    message: str
    answer: str = ""
    sources: list[str] = field(default_factory=list)
    handoff: dict[str, str] | None = None


def build_tools(turn: Turn) -> list[BaseTool]:
    """The agent's tools for one turn; they record their side effects on `turn`."""

    def search_knowledge_base(query: str) -> str:
        passages = retrieval.search(query)
        if not passages:
            return "No relevant passages found for this query."
        for passage in passages:
            if passage.condition and passage.condition not in turn.sources:
                turn.sources.append(passage.condition)
        return "\n\n".join(
            f"[{number}] ({passage.condition or 'unknown condition'}) {passage.content}"
            for number, passage in enumerate(passages, 1)
        )

    def list_conditions() -> str:
        return "The knowledge base covers: " + ", ".join(insights.condition_names())

    def request_human_handoff(reason: str, urgency: str) -> str:
        request_id = str(handoffs.create(turn.session_id, urgency, reason, turn.message))
        turn.handoff = {"id": request_id, "urgency": urgency}
        return f"Handoff request {request_id} opened with urgency {urgency}. A person from the care team will follow up."

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


async def stream_run(request: ChatRequest) -> AsyncIterator[dict[str, Any]]:
    turn = Turn(request.session_id, request.query)
    memory = PostgresChatHistory(request.session_id)

    if EMERGENCY_PATTERN.search(request.query):
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


async def run(request: ChatRequest) -> dict[str, Any]:
    """The non-streaming variant: the answer, the tools called and the handoff, if any."""
    result: dict[str, Any] = {"response": "", "sources": [], "tools": [], "handoff": None}
    async for event in stream_run(request):
        if "tool" in event:
            result["tools"].append({"name": event["tool"], "args": event["args"]})
        elif "handoff" in event:
            result["handoff"] = event["handoff"]
        elif "answer" in event:
            result["response"], result["sources"] = event["answer"], event["sources"]
    return result


async def _escalate_emergency(turn: Turn) -> AsyncIterator[dict[str, Any]]:
    args = {"reason": "Emergency wording detected in the visitor's message.", "urgency": "emergency"}
    yield {"tool": "request_human_handoff", "args": args}
    request_id = await asyncio.to_thread(handoffs.create, turn.session_id, "emergency", args["reason"], turn.message)
    turn.handoff = {"id": str(request_id), "urgency": "emergency"}
    logger.warning("Emergency guardrail opened handoff %s for session %s", request_id, turn.session_id)
    yield {"handoff": turn.handoff}
    turn.answer = EMERGENCY_ANSWER.format(id=request_id)


async def _loop(turn: Turn, history: list[BaseMessage]) -> AsyncIterator[dict[str, Any]]:
    tools = build_tools(turn)
    by_name = {tool.name: tool for tool in tools}
    model = with_tools(tools)
    messages: list[BaseMessage] = [SystemMessage(AGENT_SYSTEM_PROMPT), *history, HumanMessage(turn.message)]

    for step in range(1, MAX_STEPS + 1):
        response = await model.ainvoke(messages)
        messages.append(response)
        if not response.tool_calls:
            turn.answer = text_of(response.content).strip()
            logger.info("Agent answered after %d step(s) [model=%s]", step, model_name_of(response))
            return
        for call in response.tool_calls:
            yield {"tool": call["name"], "args": call["args"]}
            handoff_before = turn.handoff
            result = await _call_tool(by_name, call)
            messages.append(ToolMessage(result, tool_call_id=call["id"], name=call["name"]))
            if turn.handoff is not handoff_before:
                yield {"handoff": turn.handoff}

    logger.warning("Agent hit the %d-step limit for session %s", MAX_STEPS, turn.session_id)
    turn.answer = STEP_LIMIT_ANSWER


async def _call_tool(by_name: dict[str, BaseTool], call: dict[str, Any]) -> str:
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
