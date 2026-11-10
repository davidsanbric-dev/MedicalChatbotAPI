import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.core.exceptions import GENERIC_MESSAGE, RetrievalError
from app.main import app
from app.schemas import ChatRequest
from app.services import chat, insights, retrieval
from app.services.retrieval import Passage


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


@pytest.mark.parametrize(
    "body",
    [
        {"query": "   ", "session_id": "s"},
        {"query": "x" * 1001, "session_id": "s"},
        {"query": "How is acne treated?"},
    ],
)
def test_chat_request_rejects_empty_long_or_anonymous_questions(body):
    with pytest.raises(ValidationError):
        ChatRequest(**body)


def test_unwrap_json_answer():
    assert chat.unwrap_json_answer('{"response": "Rest helps."}') == "Rest helps."
    assert chat.unwrap_json_answer("Plain text.") == "Plain text."
    assert chat.unwrap_json_answer("42") == "42"


async def collect(request):
    return [event async for event in chat.stream_reply(request)]


@pytest.mark.anyio
async def test_a_turn_reports_each_stage_then_the_answer(monkeypatch):
    async def rephrase(request):
        return "acne treatment"

    async def stream_answer(request, context):
        assert context == "First passage.\n\nSecond passage."
        yield {"token": "Use "}
        yield {"answer": "Use retinoids."}

    monkeypatch.setattr(chat, "_rephrase", rephrase)
    monkeypatch.setattr(chat, "_stream_answer", stream_answer)
    monkeypatch.setattr(retrieval, "search", lambda query: [Passage("First passage.", 0.9, 3.0), Passage("Second passage.", 0.8, 1.0)])

    events = await collect(ChatRequest(query="How is acne treated?", session_id="s"))

    assert events == [
        {"stage": "rephrasing"},
        {"stage": "retrieving"},
        {"stage": "found", "count": 2},
        {"token": "Use "},
        {"answer": "Use retinoids."},
    ]


@pytest.mark.anyio
async def test_an_off_topic_question_is_answered_without_context(monkeypatch):
    contexts = []

    async def rephrase(request):
        return request.query

    async def stream_answer(request, context):
        contexts.append(context)
        yield {"answer": "I don't have that information."}

    monkeypatch.setattr(chat, "_rephrase", rephrase)
    monkeypatch.setattr(chat, "_stream_answer", stream_answer)
    monkeypatch.setattr(retrieval, "search", lambda query: [])

    events = await collect(ChatRequest(query="Who won the world cup?", session_id="s"))

    assert {"stage": "found", "count": 0} in events
    assert contexts == [chat.NO_CONTEXT]


def test_search_keeps_the_top_passages_above_the_threshold(monkeypatch):
    rows = [SimpleNamespace(content=f"p{i}", similarity=0.5) for i in range(8)]
    scores = [-5.0, 4.0, 1.0, -1.9, 3.0, 2.0, 0.5, -2.0]
    monkeypatch.setattr(retrieval, "embed_query", lambda query: [0.0])
    monkeypatch.setattr(retrieval.documents, "nearest", lambda vector, limit: rows)
    monkeypatch.setattr(retrieval, "relevance_scores", lambda query, passages: scores)

    selected = retrieval.search("q")

    assert [p.content for p in selected] == ["p1", "p4", "p5", "p2", "p6"]  # top 5, best first


def test_stream_errors_reach_the_visitor_as_a_safe_event(client, monkeypatch):
    async def failing_stream(request):
        yield {"stage": "rephrasing"}
        raise RetrievalError("Couldn't search the knowledge base.")

    monkeypatch.setattr(chat, "stream_reply", failing_stream)

    response = client.post("/chat/stream", json={"query": "q", "session_id": "s"})
    events = [json.loads(line.removeprefix("data: ")) for line in response.text.splitlines() if line]

    assert events == [{"stage": "rephrasing"}, {"error": "Couldn't search the knowledge base."}]


def test_unexpected_errors_hide_their_details(client, monkeypatch):
    def broken(limit):
        raise RuntimeError("connection to server at 10.0.0.5 failed")

    monkeypatch.setattr(insights, "top_symptoms", broken)

    response = client.get("/insights/top-symptoms")

    assert response.status_code == 500
    assert response.json() == {"detail": GENERIC_MESSAGE}


def test_top_n_is_bounded(client):
    assert client.get("/insights/top-symptoms?top_n=0").status_code == 422
    assert client.get("/insights/top-symptoms?top_n=101").status_code == 422
