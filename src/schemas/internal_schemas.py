from dataclasses import dataclass, field


@dataclass(slots=True, kw_only=True)
class EventData:
    """Data structure for event data in streaming sessions."""

    event_type: str = field(metadata={"description": "The type of the event."})
    data: dict[str, str] = field(
        metadata={"description": "The data associated with the event."}
    )
    timestamp: str = field(metadata={"description": "The timestamp of the event."})
    metadata: dict[str, str] | None = field(
        default=None,
        metadata={"description": "Additional metadata for the event, if any."},
    )
