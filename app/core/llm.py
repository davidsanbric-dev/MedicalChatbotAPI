"""The chat model: OpenRouter first, Gemini as automatic fallback.

See "Provider failover" in https://medicalchatbot.demos.davieldev.com/api#notes
"""

import json
from collections.abc import Sequence

from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from app.core.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
    OPENROUTER_REASONING_EFFORT,
)

# Gemini acts as the fallback provider.
gemini_llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    google_api_key=GEMINI_API_KEY,
)

# OpenRouter (OpenAI-compatible) is the default/primary provider when configured.
# If a call to OpenRouter fails (auth, rate limit, outage, removed model, ...),
# LangChain sends the same request to Gemini via `with_fallbacks`. That fallback
# is the retry: retrying a busy free model first only delays the answer.
openrouter_llm = None
if OPENROUTER_API_KEY:
    openrouter_llm = ChatOpenAI(
        model=OPENROUTER_MODEL,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        max_retries=0,
        extra_body={"reasoning": {"effort": OPENROUTER_REASONING_EFFORT}},
    )
    llm = openrouter_llm.with_fallbacks([gemini_llm])
else:
    # No OpenRouter key configured: fall back to Gemini as the sole provider.
    llm = gemini_llm


def with_tools(tools: Sequence[BaseTool]) -> Runnable:
    """The chat model able to call `tools`, with the same Gemini fallback as `llm`.

    Tools are bound to each provider before `with_fallbacks`: the fallback wrapper
    itself has no bind_tools, and Gemini must know the tools too when it answers.
    """
    if openrouter_llm is None:
        return gemini_llm.bind_tools(tools)
    return openrouter_llm.bind_tools(tools).with_fallbacks([gemini_llm.bind_tools(tools)])


def text_of(content: str | list) -> str:
    """Message content as plain text; some providers send a list of content parts."""
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)


def unwrap_json_answer(text: str) -> str:
    """Some models wrap the whole answer in JSON, like {"response": "..."}: keep only the text."""
    try:
        parsed = json.loads(text)
    except ValueError:
        return text
    if isinstance(parsed, dict):
        return parsed.get("response") or parsed.get("content") or parsed.get("detail") or text
    return text


def model_name_of(message) -> str:
    """Best-effort extraction of the model that actually produced a response.

    OpenRouter may route a request to a different underlying model than the one
    requested, and the Gemini fallback may answer instead, so we read the model
    reported in the response metadata rather than assuming OPENROUTER_MODEL.
    """
    metadata = getattr(message, "response_metadata", None) or {}
    return metadata.get("model_name") or metadata.get("model") or "unknown"
