"""Agent evaluation: does the agent pick the right tool for each message?

Run with `make eval-agent` (needs the local database and the LLM keys in .env; about two
model calls per case). Writes evals/results/agent.md.

Each case in agent_cases.jsonl names what should happen:
    search_knowledge_base | list_conditions | request_human_handoff   that tool is called
    none        a plain answer, no tool (greetings, thanks)
    guardrail   the emergency guardrail escalates before the model is asked

Every run uses fresh "eval-agent-" sessions and deletes their chat history and handoff
requests afterwards, so the evaluation leaves no tickets behind.
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import date
from pathlib import Path

from app.core.config import GEMINI_MODEL, OPENROUTER_API_KEY, OPENROUTER_MODEL
from app.core.database import prepare_database
from app.dependencies import get_agent_service
from app.domain.triage import is_emergency
from app.repositories.chat_history import ChatHistoryRepository
from app.repositories.handoffs import HandoffRepository
from app.schemas import ChatRequest

HERE = Path(__file__).parent
SESSION_PREFIX = "eval-agent-"
PAUSE_SECONDS = 2  # between cases, to stay under free-tier rate limits


def outcome(message: str, tools: list[str]) -> str:
    """What the agent actually did, in the labels of agent_cases.jsonl."""
    if is_emergency(message):
        return "guardrail"
    return tools[0] if tools else "none"


async def run_case(message: str) -> dict:
    session_id = SESSION_PREFIX + uuid.uuid4().hex[:12]
    started = time.perf_counter()
    try:
        result = await get_agent_service().reply(ChatRequest(query=message, session_id=session_id))
        tools = [tool["name"] for tool in result.tools]
        return {
            "session_id": session_id,
            "tools": tools,
            "did": outcome(message, tools),
            "answer": result.response,
            "seconds": time.perf_counter() - started,
        }
    except Exception as error:  # a failed case is a result, not a crash
        return {"session_id": session_id, "tools": [], "did": f"error: {type(error).__name__}", "answer": str(error), "seconds": 0.0}


async def evaluate(cases: list[dict]) -> list[dict]:
    rows = []
    for number, case in enumerate(cases, 1):
        row = {**case, **await run_case(case["message"])}
        row["passed"] = row["did"] == case["expected"]
        rows.append(row)
        print(f"[{number}/{len(cases)}] {'PASS' if row['passed'] else 'FAIL'} {case['message']!r} -> {row['did']}")
        await asyncio.sleep(PAUSE_SECONDS)
    return rows


def cleanup(rows: list[dict]) -> None:
    chat_history_repository = ChatHistoryRepository()
    for row in rows:
        chat_history_repository.clear(row["session_id"])
    removed = HandoffRepository().delete_by_session_prefix(SESSION_PREFIX)
    print(f"Cleaned up {len(rows)} eval sessions and {removed} handoff requests.")


def report(rows: list[dict]) -> str:
    passed = sum(row["passed"] for row in rows)
    model = f"{OPENROUTER_MODEL} via OpenRouter, Gemini {GEMINI_MODEL} as fallback" if OPENROUTER_API_KEY else f"Gemini {GEMINI_MODEL}"
    by_expected: dict[str, list[bool]] = {}
    for row in rows:
        by_expected.setdefault(row["expected"], []).append(row["passed"])
    timed = [row["seconds"] for row in rows if row["seconds"]]
    median_turn = f"{sorted(timed)[len(timed) // 2]:.1f} s" if timed else "n/a (every case failed)"
    false_handoffs = sum(row["did"] == "request_human_handoff" and row["expected"] != "request_human_handoff" for row in rows)
    lines = [
        f"# Agent evaluation ({date.today().isoformat()})",
        "",
        (
            f"Model: {model}. {passed} of {len(rows)} cases chose the expected action ({passed / len(rows):.0%}); "
            f"{false_handoffs} unnecessary handoffs; median turn {median_turn}."
        ),
        "",
        "| Expected action | Correct |",
        "| --- | --- |",
        *(f"| {expected} | {sum(results)}/{len(results)} |" for expected, results in by_expected.items()),
        "",
        "| Message | Expected | Did | Tools called | Seconds |",
        "| --- | --- | --- | --- | --- |",
        *(
            f"| {row['message']} | {row['expected']} | {'✓ ' if row['passed'] else '✗ '}{row['did']} "
            f"| {', '.join(row['tools']) or '-'} | {row['seconds']:.1f} |"
            for row in rows
        ),
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    prepare_database()  # creates handoff_requests if the API hasn't run since it was added
    cases = [json.loads(line) for line in (HERE / "agent_cases.jsonl").read_text().splitlines() if line.strip()]
    rows = asyncio.run(evaluate(cases))
    cleanup(rows)
    text = report(rows)
    out = HERE / "results" / "agent.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(text)
    # The same results for the website (medical_conditions_react_frontend/src/data/evaluation/agent.json).
    data = {
        "date": date.today().isoformat(),
        "model": OPENROUTER_MODEL if OPENROUTER_API_KEY else GEMINI_MODEL,
        "fallback": GEMINI_MODEL if OPENROUTER_API_KEY else None,
        "cases": [
            {key: row[key] for key in ("message", "expected", "did", "passed", "tools")} | {"seconds": round(row["seconds"], 1)}
            for row in rows
        ],
    }
    (HERE / "results" / "agent.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(text)
    print(f"Written to {out.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
