"""Settings from environment variables (or a local .env file)."""

import os

from dotenv import load_dotenv

load_dotenv()

# --- Database ---
DATABASE_URL = os.getenv("DATABASE_URL")

# --- Gemini (fallback provider) ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# --- OpenRouter (default/primary provider, OpenAI-compatible) ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-120b:free")

# --- Rate limiting (chat endpoints; 0 disables a limit) ---
# Per client, so one visitor can't use up the LLM quota...
CHAT_RATE_LIMIT_PER_MINUTE = int(os.getenv("CHAT_RATE_LIMIT_PER_MINUTE", "6"))
CHAT_RATE_LIMIT_PER_DAY = int(os.getenv("CHAT_RATE_LIMIT_PER_DAY", "60"))
# ...and across all clients, which bounds spend even if clients rotate IPs.
CHAT_GLOBAL_LIMIT_PER_DAY = int(os.getenv("CHAT_GLOBAL_LIMIT_PER_DAY", "500"))
# How many reverse proxies in front of the API append to X-Forwarded-For.
# 0 keys clients by the direct peer address (safe, but behind a proxy every
# visitor then shares one bucket). See README "Rate Limiting".
TRUSTED_PROXY_HOPS = int(os.getenv("TRUSTED_PROXY_HOPS", "0"))
