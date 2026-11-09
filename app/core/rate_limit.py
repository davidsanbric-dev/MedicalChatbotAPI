"""In-memory sliding-window rate limiting for the chat endpoints.

Every chat turn costs two LLM calls, so the chat routes are limited per client
(per minute and per day) and globally per day. State lives in this process,
which matches how the API runs (a single uvicorn process); running several
replicas would need a shared store such as Redis instead.
"""

import logging
import math
import time
from collections import deque

from fastapi import HTTPException, Request

from app.core.config import (
    CHAT_GLOBAL_LIMIT_PER_DAY,
    CHAT_RATE_LIMIT_PER_DAY,
    CHAT_RATE_LIMIT_PER_MINUTE,
    TRUSTED_PROXY_HOPS,
)

logger = logging.getLogger(__name__)

MINUTE = 60
DAY = 24 * 60 * 60


class SlidingWindowLimiter:
    """Allow at most `limit` hits per key within any `window` seconds."""

    def __init__(self, limit: int, window: float, message: str, clock=time.monotonic):
        self.limit = limit
        self.window = window
        self.message = message
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}

    def retry_after(self, key: str) -> float:
        """Seconds until `key` may make another request (0 if allowed now)."""
        if self.limit <= 0:
            return 0.0
        now = self._clock()
        hits = self._hits.get(key)
        if hits is None:
            return 0.0
        while hits and hits[0] <= now - self.window:
            hits.popleft()
        if not hits:
            del self._hits[key]
            return 0.0
        if len(hits) < self.limit:
            return 0.0
        return hits[0] + self.window - now

    def hit(self, key: str) -> None:
        if self.limit > 0:
            self._hits.setdefault(key, deque()).append(self._clock())

    def sweep(self) -> None:
        """Forget keys with no hits inside the window, so memory stays bounded."""
        cutoff = self._clock() - self.window
        for key in [k for k, hits in self._hits.items() if not hits or hits[-1] <= cutoff]:
            del self._hits[key]


def client_ip(request: Request, trusted_hops: int = TRUSTED_PROXY_HOPS) -> str:
    """The caller's address, looking through `trusted_hops` reverse proxies.

    Each proxy appends the address it received the request from to
    X-Forwarded-For, so with N trusted proxies the client is the Nth address
    from the right of [*X-Forwarded-For, direct peer]. Entries further left are
    supplied by the client and can be forged, so they are never used.
    """
    peer = request.client.host if request.client else "unknown"
    if trusted_hops <= 0:
        return peer
    forwarded = [part.strip() for part in request.headers.get("x-forwarded-for", "").split(",") if part.strip()]
    chain = [*forwarded, peer]
    return chain[-(trusted_hops + 1)] if len(chain) > trusted_hops else chain[0]


GLOBAL_KEY = "*"

_per_client = [
    SlidingWindowLimiter(
        CHAT_RATE_LIMIT_PER_MINUTE, MINUTE, "Too many questions in a short time. Please wait a moment."
    ),
    SlidingWindowLimiter(CHAT_RATE_LIMIT_PER_DAY, DAY, "You've reached today's question limit for this demo."),
]
_global = SlidingWindowLimiter(
    CHAT_GLOBAL_LIMIT_PER_DAY, DAY, "The demo has reached its daily question limit. Please come back tomorrow."
)
_calls = 0


async def limit_chat(request: Request) -> None:
    """FastAPI dependency: reject the request with 429 when any limit is reached.

    All limits are checked before any is recorded, so a rejected request never
    uses up quota.
    """
    global _calls
    key = client_ip(request)
    checks = [(limiter, key) for limiter in _per_client] + [(_global, GLOBAL_KEY)]

    blocked = [(limiter.retry_after(k), limiter) for limiter, k in checks]
    wait, limiter = max(blocked, key=lambda pair: pair[0])
    if wait > 0:
        logger.warning("Rate limit hit for %s (%d per %ds)", key, limiter.limit, limiter.window)
        raise HTTPException(
            status_code=429,
            detail=limiter.message,
            headers={"Retry-After": str(math.ceil(wait))},
        )

    for limiter, k in checks:
        limiter.hit(k)

    _calls += 1
    if _calls % 1000 == 0:
        for limiter, _ in checks:
            limiter.sweep()
