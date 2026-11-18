"""Server-Sent Events for the chat and agent streams."""

import json
import logging
from collections.abc import AsyncIterator, Mapping
from typing import Any

from fastapi.responses import StreamingResponse

from app.core.exceptions import public_message
from app.schemas import ErrorEvent

logger = logging.getLogger(__name__)


class EventStream(StreamingResponse):
    """A Server-Sent Events response; as `response_class`, it also labels the route in /docs."""

    media_type = "text/event-stream"


def sse_response(events: AsyncIterator[Mapping[str, Any]], session_id: str) -> EventStream:
    """One `data: <json>` line per event."""

    async def lines():
        # Once streaming starts the status code is already sent, so a failure goes out
        # as a final event instead of through the app's exception handler.
        try:
            async for event in events:
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as error:
            logger.exception("Stream failed for session %s", session_id)
            failure: ErrorEvent = {"error": public_message(error)}
            yield f"data: {json.dumps(failure)}\n\n"

    return EventStream(lines(), headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
