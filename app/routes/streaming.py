"""Server-Sent Events for the chat and agent streams."""

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from fastapi.responses import StreamingResponse

from app.core.exceptions import public_message

logger = logging.getLogger(__name__)


def sse_response(events: AsyncIterator[dict[str, Any]], session_id: str) -> StreamingResponse:
    """One `data: <json>` line per event."""

    async def lines():
        # Once streaming starts the status code is already sent, so a failure goes out
        # as a final event instead of through the app's exception handler.
        try:
            async for event in events:
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as error:
            logger.exception("Stream failed for session %s", session_id)
            yield f"data: {json.dumps({'error': public_message(error)})}\n\n"

    return StreamingResponse(
        lines(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
