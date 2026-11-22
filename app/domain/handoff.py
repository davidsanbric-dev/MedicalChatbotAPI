"""A conversation handed to a human from the care team: the stand-in for a help-desk ticket."""

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum


class Urgency(StrEnum):
    ROUTINE = "routine"
    URGENT = "urgent"  # care needed today
    EMERGENCY = "emergency"  # life-threatening signs


EMERGENCY_REASON = "Emergency wording detected in the visitor's message."


@dataclass(frozen=True)
class Handoff:
    """An entity: its id is the ticket number the visitor and the care team see, generated once
    and never changed (the dataclass is frozen). Two handoffs are the same request when their
    ids match."""

    session_id: str = field(compare=False)
    urgency: Urgency = field(compare=False)
    reason: str = field(compare=False)
    last_message: str = field(compare=False)  # what the visitor said when it was raised
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    opened_at: datetime = field(default_factory=lambda: datetime.now(UTC), compare=False)

    @property
    def reference(self) -> str:
        """The short request number told to the visitor: the id's first 8 hex digits."""
        return self.id.hex[:8]

    @classmethod
    def for_emergency(cls, session_id: str, last_message: str) -> "Handoff":
        """The handoff the emergency guardrail opens, before any model is asked."""
        return cls(session_id, Urgency.EMERGENCY, EMERGENCY_REASON, last_message)
