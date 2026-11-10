"""The chat model: OpenRouter first, Gemini as automatic fallback.

See "Provider failover" in https://medicalchatbot.demos.davieldev.com/api#notes
"""

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


def model_name_of(message) -> str:
    """Best-effort extraction of the model that actually produced a response.

    OpenRouter may route a request to a different underlying model than the one
    requested, and the Gemini fallback may answer instead, so we read the model
    reported in the response metadata rather than assuming OPENROUTER_MODEL.
    """
    metadata = getattr(message, "response_metadata", None) or {}
    return metadata.get("model_name") or metadata.get("model") or "unknown"
