"""The agent loop with a scripted model: no database, network or API keys."""

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from app.domain.passages import Passage
from app.schemas import ChatRequest
from app.services import agent
from app.services.agent import AgentService
from tests.conftest import StubRetrievalService


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


class ListHandoffRepository:
    """Stands in for HandoffRepository: the opened handoffs, in a list."""

    def __init__(self):
        self.added = []

    def add(self, handoff):
        self.added.append(handoff)


class StubInsightsService:
    def condition_names(self):
        return ["Acne", "Asthma"]


@pytest.fixture
def tickets():
    return ListHandoffRepository()


@pytest.fixture
def service(chat_histories, tickets):
    """The agent with fakes for its storage and services; each test picks the passages and model."""
    return AgentService(StubRetrievalService([]), StubInsightsService(), tickets, chat_histories)


def use_model(monkeypatch, model):
    monkeypatch.setattr(agent, "with_tools", lambda tools: model)


async def collect(service, query, session_id="s"):
    return [event async for event in service.stream(ChatRequest(query=query, session_id=session_id))]


@pytest.mark.anyio
async def test_the_agent_searches_then_answers_from_the_results(monkeypatch, service):
    model = ScriptedModel(tool_call("search_knowledge_base", {"query": "acne treatment"}), AIMessage("Retinoids help (Acne)."))
    use_model(monkeypatch, model)
    service.retrieval_service = StubRetrievalService([Passage("Retinoids treat acne.", 0.9, 5.0, "Acne")])

    events = await collect(service, "How is acne treated?")

    assert events == [
        {"tool": "search_knowledge_base", "args": {"query": "acne treatment"}},
        {"answer": "Retinoids help (Acne).", "sources": ["Acne"]},
    ]
    tool_result = model.calls[1][-1]
    assert isinstance(tool_result, ToolMessage) and "(Acne) Retinoids treat acne." in tool_result.content
    # Memory keeps the conversation, not the tool traffic.
    assert [m.content for m in service.chat_history("s").messages] == ["How is acne treated?", "Retinoids help (Acne)."]


@pytest.mark.anyio
async def test_a_handoff_opens_a_request_and_is_reported(monkeypatch, service, tickets):
    args = {"reason": "Wants a prescription refill.", "urgency": "routine"}
    use_model(monkeypatch, ScriptedModel(tool_call("request_human_handoff", args), AIMessage("A person will follow up.")))

    events = await collect(service, "I need my prescription refilled")

    [ticket] = tickets.added
    assert (ticket.session_id, ticket.urgency, ticket.reason, ticket.last_message) == ("s", "routine", args["reason"], "I need my prescription refilled")
    [event] = [e for e in events if "handoff" in e]
    assert event["handoff"] == {
        "id": str(ticket.id),
        "reference": ticket.id.hex[:8],
        "urgency": "routine",
        "reason": args["reason"],
        "opened_at": ticket.opened_at.isoformat(),
    }
    assert events[-1]["answer"] == "A person will follow up."


@pytest.mark.anyio
async def test_emergency_wording_escalates_without_asking_the_model(monkeypatch, service, tickets):
    model = ScriptedModel()
    use_model(monkeypatch, model)

    events = await collect(service, "Tengo dolor de pecho muy fuerte")

    assert model.calls == []
    assert tickets.added[0].urgency == "emergency"
    assert "123 en Colombia" in events[-1]["answer"]  # the advice follows the visitor's Spanish
    assert f"(solicitud {tickets.added[0].reference})" in events[-1]["answer"]


@pytest.mark.anyio
async def test_invalid_tool_arguments_go_back_to_the_model_as_an_error(monkeypatch, service, tickets):
    bad = tool_call("request_human_handoff", {"reason": "help", "urgency": "whenever"})
    model = ScriptedModel(bad, AIMessage("Sorry, let me try again later."))
    use_model(monkeypatch, model)

    events = await collect(service, "Can I talk to someone?")

    assert tickets.added == []  # the schema rejected "whenever" before the tool ran
    assert model.calls[1][-1].content.startswith("Error: the tool request_human_handoff failed")
    assert events[-1]["answer"] == "Sorry, let me try again later."


@pytest.mark.anyio
async def test_a_model_that_keeps_calling_tools_is_stopped(monkeypatch, service):
    calls = [tool_call("search_knowledge_base", {"query": "acne"}, f"call-{i}") for i in range(agent.MAX_STEPS)]
    use_model(monkeypatch, ScriptedModel(*calls))

    events = await collect(service, "What do you cover?")

    assert sum("tool" in event for event in events) == agent.MAX_STEPS
    assert events[-1]["answer"] == agent.STEP_LIMIT_ANSWER


@pytest.mark.anyio
async def test_the_listed_conditions_go_to_the_frontend_whole_and_the_model_introduces_them(monkeypatch, service):
    model = ScriptedModel(tool_call("list_conditions", {}), AIMessage("I cover 2 conditions; ask me about any of them."))
    use_model(monkeypatch, model)

    result = await service.reply(ChatRequest(query="What can you help me with?", session_id="s"))

    assert result.conditions == ["Acne", "Asthma"]  # the whole list, as data, whatever the model writes
    assert result.response == "I cover 2 conditions; ask me about any of them."
    assert "don't repeat them" in model.calls[1][-1].content  # the model is told the visitor sees the list
    # The history keeps the names, so a follow-up about "the first one" has them.
    assert service.chat_history("s").messages[-1].content.endswith("Acne, Asthma")


@pytest.mark.anyio
async def test_conditions_listed_alongside_another_tool_still_get_an_answer_from_the_model(monkeypatch, service):
    both = AIMessage("", tool_calls=[
        {"name": "list_conditions", "args": {}, "id": "call-1"},
        {"name": "search_knowledge_base", "args": {"query": "acne"}, "id": "call-2"},
    ])
    model = ScriptedModel(both, AIMessage("Here is what I cover, and about acne…"))
    use_model(monkeypatch, model)

    events = await collect(service, "What do you cover, and what is acne?")

    assert {"conditions": ["Acne", "Asthma"]} in events
    assert events[-1]["answer"] == "Here is what I cover, and about acne…"
    assert "don't repeat them" in model.calls[1][-2].content  # the list_conditions result


@pytest.mark.anyio
async def test_greetings_need_no_tools(monkeypatch, service):
    use_model(monkeypatch, ScriptedModel(AIMessage("Hi! How can I help?")))

    result = await service.reply(ChatRequest(query="hello", session_id="s"))

    assert result.model_dump() == {"response": "Hi! How can I help?", "sources": [], "tools": [], "handoff": None, "conditions": None}


@pytest.mark.anyio
async def test_an_answer_wrapped_in_json_is_unwrapped(monkeypatch, service):
    use_model(monkeypatch, ScriptedModel(AIMessage('{"response": "Hi! How can I help?"}')))

    events = await collect(service, "hello")

    assert events[-1]["answer"] == "Hi! How can I help?"


@pytest.mark.anyio
async def test_the_non_streaming_reply_lists_the_tools_and_the_handoff(monkeypatch, service, tickets):
    args = {"reason": "Wants a prescription refill.", "urgency": "routine"}
    use_model(monkeypatch, ScriptedModel(tool_call("request_human_handoff", args), AIMessage("A person will follow up.")))

    result = await service.reply(ChatRequest(query="I need my prescription refilled", session_id="s"))

    assert result.model_dump(mode="json") == {
        "response": "A person will follow up.",
        "sources": [],
        "tools": [{"name": "request_human_handoff", "args": args}],
        "handoff": {
            "id": str(tickets.added[0].id),
            "reference": tickets.added[0].reference,
            "urgency": "routine",
            "reason": args["reason"],
            "opened_at": tickets.added[0].opened_at.isoformat(),
        },
        "conditions": None,
    }
