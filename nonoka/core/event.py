from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol, runtime_checkable
from pydantic import BaseModel, Field


class EventType(str, Enum):
  """Standardize built-in event types"""
  # Session level
  SESSION_STARTED = "session.started"
  SESSION_COMPLETED = "session.completed"
  SESSION_FAILED = "session.failed"

  # Plan and execution level
  PLAN_GENERATED = "plan.generated"
  STEP_STARTED = "step.started"
  STEP_COMPLETED = "step.completed"
  STEP_FAILED = "step.failed"

  # LLM and tool level
  LLM_CALLED = "llm.called"
  TOOL_CALLED = "tool.called"
  TOOL_COMPLETED = "tool.completed"


class AgentEvent(BaseModel):
  """
  Framework-issued structured observability event.
  All telemetry data and run logs are distributed based on this structure.
  """
  type: EventType | str
  session_id: str
  timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
  data: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class ObservabilityBackend(Protocol):
  """
  Event consumption protocol.
  Used for observability.
  """
  async def on_event(self, event: AgentEvent) -> None:
    ...