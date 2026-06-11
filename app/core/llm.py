from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from app.core.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
)

# Gemini acts as the fallback provider.
gemini_llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    google_api_key=GEMINI_API_KEY,
)

# OpenRouter (OpenAI-compatible) is the default/primary provider when configured.
# If a call to OpenRouter fails (auth, rate limit, outage, ...), LangChain
# automatically retries the same request with Gemini via `with_fallbacks`.
if OPENROUTER_API_KEY:
    openrouter_llm = ChatOpenAI(
        model=OPENROUTER_MODEL,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )
    llm = openrouter_llm.with_fallbacks([gemini_llm])
else:
    # No OpenRouter key configured: fall back to Gemini as the sole provider.
    llm = gemini_llm
