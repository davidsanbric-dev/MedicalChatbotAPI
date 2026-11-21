"""The tool-calling agent: the model decides, turn by turn, whether to search the knowledge
base, list what it covers, or hand the conversation to a human. Acted out in 3D, one step
at a time, on https://medicalchatbot.demos.davieldev.com/api

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
from app.domain.passages import Passage
from app.domain.triage import EMERGENCY_ADVICE, is_emergency
from app.repositories.chat_history import ChatHistoryRepository
from app.repositories.handoffs import HandoffRepository
from app.schemas import AgentEvent, AgentResponse, ChatRequest, HandoffEvent
from app.services.base import ConversationService
from app.services.insights import InsightsService
from app.services.retrieval import RetrievalService

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
    """One visitor message and what the agent did about it. The tools fill it in as they run;
    the answer, its sources and the handoff are then read from it."""

    session_id: str
    message: str
    answer: str = ""
    sources: list[str] = field(default_factory=list)  # conditions the answer drew on
    handoff: Handoff | None = None

    def add_sources(self, passages: list[Passage]) -> None:
        """Remember the conditions these passages came from, once each, in the order found."""
        for passage in passages:
            if passage.condition and passage.condition not in self.sources:
                self.sources.append(passage.condition)


class AgentTools:
    """The tools the model may call during one turn. Each tool is a method whose docstring is
    the description the model reads; what a tool does is also recorded on the turn."""

    def __init__(
        self,
        turn: Turn,
        retrieval_service: RetrievalService,
        insights_service: InsightsService,
        handoff_repository: HandoffRepository,
    ) -> None:
        self.turn = turn
        self.retrieval_service = retrieval_service
        self.insights_service = insights_service
        self.handoff_repository = handoff_repository
        self.for_model: list[BaseTool] = [
            StructuredTool.from_function(self.search_knowledge_base, args_schema=SearchArgs),
            StructuredTool.from_function(self.list_conditions),
            StructuredTool.from_function(self.request_human_handoff, args_schema=HandoffArgs),
        ]
        self._by_name = {tool.name: tool for tool in self.for_model}

    def search_knowledge_base(self, query: str) -> str:
        """Search the medical knowledge base for passages about symptoms, causes and treatments."""
        passages = self.retrieval_service.search(query)
        self.turn.add_sources(passages)
        return search_results(passages)

    def list_conditions(self) -> str:
        """List the conditions the knowledge base covers."""
        return "The knowledge base covers: " + ", ".join(self.insights_service.condition_names())

    def request_human_handoff(self, reason: str, urgency: Urgency) -> str:
        """Hand the conversation to a human from the care team."""
        handoff = Handoff(self.turn.session_id, Urgency(urgency), reason, self.turn.message)
        self.handoff_repository.add(handoff)
        self.turn.handoff = handoff
        return f"Handoff request {handoff.id} opened with urgency {handoff.urgency}. A person from the care team will follow up."

    async def run(self, call: ToolCall) -> ToolMessage:
        """Run one tool call and return its result as the tool message the model reads next."""
        return ToolMessage(await self._result_of(call), tool_call_id=call["id"], name=call["name"])

    async def _result_of(self, call: ToolCall) -> str:
        """The tool's text result; any failure goes back to the model as text it can react to."""
        tool = self._by_name.get(call["name"])
        if tool is None:
            return f"Error: there is no tool named {call['name']!r}."
        try:
            # Tools hit the database and the ONNX models: off the event loop, like the chat's retrieval.
            return await asyncio.to_thread(tool.invoke, call["args"])
        except Exception as error:
            logger.warning("Tool %s failed: %s", call["name"], error)
            return f"Error: the tool {call['name']} failed ({type(error).__name__}). Try different arguments or answer without it."


class AgentService(ConversationService[AgentEvent, AgentResponse]):
    def __init__(
        self,
        retrieval_service: RetrievalService,
        insights_service: InsightsService,
        handoff_repository: HandoffRepository,
        chat_history_repository: ChatHistoryRepository,
    ) -> None:
        super().__init__(chat_history_repository)
        self.retrieval_service = retrieval_service
        self.insights_service = insights_service
        self.handoff_repository = handoff_repository

    async def stream(self, request: ChatRequest) -> AsyncIterator[AgentEvent]:
        """Run the agent on the visitor's message and stream what it does."""
        turn = Turn(request.session_id, request.query)
        chat_history = self.chat_history(request.session_id)

        if is_emergency(request.query):
            events = self._escalate_emergency(turn)
        else:
            events = self._run_tool_loop(turn, await chat_history.aget_messages())
        async for event in events:
            yield event

        # Only the visitor's message and the final answer are kept: the next turn needs the
        # conversation, not this turn's tool traffic.
        await chat_history.aadd_messages([HumanMessage(request.query), AIMessage(turn.answer)])
        yield {"answer": turn.answer, "sources": turn.sources}

    async def _escalate_emergency(self, turn: Turn) -> AsyncIterator[AgentEvent]:
        """Open an emergency handoff and answer with fixed safety advice, without asking the model."""
        handoff = Handoff.for_emergency(turn.session_id, turn.message)
        yield {"tool": "request_human_handoff", "args": {"reason": handoff.reason, "urgency": handoff.urgency}}
        await asyncio.to_thread(self.handoff_repository.add, handoff)
        turn.handoff = handoff
        logger.warning("Emergency guardrail opened handoff %s for session %s", handoff.id, turn.session_id)
        yield _handoff_event(handoff)
        turn.answer = EMERGENCY_ADVICE.format(id=handoff.id)

    async def _run_tool_loop(self, turn: Turn, history: list[BaseMessage]) -> AsyncIterator[AgentEvent]:
        """Run the agent loop for a single turn: ask the model, run the tools it asks for, and
        repeat until it answers in plain text (at most MAX_STEPS model calls)."""
        tools = AgentTools(turn, self.retrieval_service, self.insights_service, self.handoff_repository)
        model = with_tools(tools.for_model)
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
                messages.append(await tools.run(call))
                if turn.handoff is not None and turn.handoff is not handoff_before:  # the tool opened one
                    yield _handoff_event(turn.handoff)

        logger.warning("Agent hit the %d-step limit for session %s", MAX_STEPS, turn.session_id)
        turn.answer = STEP_LIMIT_ANSWER

    def _new_response(self) -> AgentResponse:
        return AgentResponse(response="", sources=[], tools=[])

    def _add_to_response(self, response: AgentResponse, event: AgentEvent) -> None:
        if "tool" in event:
            response.tools.append({"name": event["tool"], "args": event["args"]})
        elif "handoff" in event:
            response.handoff = event["handoff"]
        elif "answer" in event:
            response.response, response.sources = event["answer"], event["sources"]


def _handoff_event(handoff: Handoff) -> HandoffEvent:
    return {"handoff": {"id": str(handoff.id), "urgency": handoff.urgency}}
