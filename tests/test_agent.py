"""The agent loop with a scripted model: no database, network or API keys."""

import uuid

import pytest
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, ToolMessage

from app.schemas import ChatRequest
from app.services import agent, retrieval
from app.services.retrieval import Passage


class ScriptedModel:
    """Returns the given replies in order and records the messages it was sent."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    async def ainvoke(self, messages):
        self.calls.append(list(messages))
        return self.replies.pop(0)


def tool_call(name, args, call_id="call-1"):
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": call_id}])


@pytest.fixture
def memory(monkeypatch):
    histories = {}
    monkeypatch.setattr(agent, "PostgresChatHistory", lambda sid: histories.setdefault(sid, InMemoryChatMessageHistory()))
    return histories


@pytest.fixture
def tickets(monkeypatch):
    created = []

    def create(session_id, urgency, reason, last_message):
        created.append({"session_id": session_id, "urgency": urgency, "reason": reason, "last_message": last_message})
        return uuid.UUID(int=len(created))

    monkeypatch.setattr(agent.handoffs, "create", create)
    return created


def use_model(monkeypatch, model):
    monkeypatch.setattr(agent, "with_tools", lambda tools: model)


async def collect(query, session_id="s"):
    return [event async for event in agent.stream_run(ChatRequest(query=query, session_id=session_id))]


@pytest.mark.anyio
async def test_the_agent_searches_then_answers_from_the_results(monkeypatch, memory):
    model = ScriptedModel(tool_call("search_knowledge_base", {"query": "acne treatment"}), AIMessage("Retinoids help (Acne)."))
    use_model(monkeypatch, model)
    monkeypatch.setattr(retrieval, "search", lambda query: [Passage("Retinoids treat acne.", 0.9, 5.0, "Acne")])

    events = await collect("How is acne treated?")

    assert events == [
        {"tool": "search_knowledge_base", "args": {"query": "acne treatment"}},
        {"answer": "Retinoids help (Acne).", "sources": ["Acne"]},
    ]
    tool_result = model.calls[1][-1]
    assert isinstance(tool_result, ToolMessage) and "(Acne) Retinoids treat acne." in tool_result.content
    # Memory keeps the conversation, not the tool traffic.
    assert [m.content for m in memory["s"].messages] == ["How is acne treated?", "Retinoids help (Acne)."]


@pytest.mark.anyio
async def test_a_handoff_opens_a_request_and_is_reported(monkeypatch, memory, tickets):
    args = {"reason": "Wants a prescription refill.", "urgency": "routine"}
    use_model(monkeypatch, ScriptedModel(tool_call("request_human_handoff", args), AIMessage("A person will follow up.")))

    events = await collect("I need my prescription refilled")

    assert tickets == [{"session_id": "s", "urgency": "routine", "reason": args["reason"], "last_message": "I need my prescription refilled"}]
    assert {"handoff": {"id": str(uuid.UUID(int=1)), "urgency": "routine"}} in events
    assert events[-1]["answer"] == "A person will follow up."


@pytest.mark.anyio
async def test_emergency_wording_escalates_without_asking_the_model(monkeypatch, memory, tickets):
    model = ScriptedModel()
    use_model(monkeypatch, model)

    events = await collect("Tengo dolor de pecho muy fuerte")

    assert model.calls == []
    assert tickets[0]["urgency"] == "emergency"
    assert "123 in Colombia" in events[-1]["answer"]


@pytest.mark.parametrize(
    "message",
    ["I've been thinking about ending my life", "I want to kill myself", "Mi hijo está inconsciente", "No puedo respirar"],
)
def test_emergency_wording_is_recognized(message):
    assert agent.EMERGENCY_PATTERN.search(message)


@pytest.mark.parametrize("message", ["My chest feels congested from a cold", "How long does the flu last?"])
def test_ordinary_questions_are_not_emergencies(message):
    assert not agent.EMERGENCY_PATTERN.search(message)


@pytest.mark.anyio
async def test_invalid_tool_arguments_go_back_to_the_model_as_an_error(monkeypatch, memory, tickets):
    bad = tool_call("request_human_handoff", {"reason": "help", "urgency": "whenever"})
    model = ScriptedModel(bad, AIMessage("Sorry, let me try again later."))
    use_model(monkeypatch, model)

    events = await collect("Can I talk to someone?")

    assert tickets == []  # the schema rejected "whenever" before the tool ran
    assert model.calls[1][-1].content.startswith("Error: the tool request_human_handoff failed")
    assert events[-1]["answer"] == "Sorry, let me try again later."


@pytest.mark.anyio
async def test_a_model_that_keeps_calling_tools_is_stopped(monkeypatch, memory):
    calls = [tool_call("list_conditions", {}, f"call-{i}") for i in range(agent.MAX_STEPS)]
    use_model(monkeypatch, ScriptedModel(*calls))
    monkeypatch.setattr(agent.insights, "condition_names", lambda: ["Acne"])

    events = await collect("What do you cover?")

    assert sum("tool" in event for event in events) == agent.MAX_STEPS
    assert events[-1]["answer"] == agent.STEP_LIMIT_ANSWER


@pytest.mark.anyio
async def test_greetings_need_no_tools(monkeypatch, memory):
    use_model(monkeypatch, ScriptedModel(AIMessage("Hi! How can I help?")))

    result = await agent.run(ChatRequest(query="hello", session_id="s"))

    assert result == {"response": "Hi! How can I help?", "sources": [], "tools": [], "handoff": None}
