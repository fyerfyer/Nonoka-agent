from nonoka.core.plan import Plan
from enum import Enum
from typing import Any, Protocol, runtime_checkable
from pydantic import BaseModel, Field
from datetime import datetime


class SessionStatus(str, Enum):
  """Session status"""
  CREATED = "created"
  RUNNING = "running"
  PAUSED = "paused"
  COMPLETED = "completed"
  FAILED = "failed"


class StepStatus(str, Enum):
  """Step status"""
  PENDING = "pending"
  RUNNING = "running"
  COMPLETED = "completed"
  FAILED = "failed"


class StepResult(BaseModel):
  """Tool execution success result for easy serialization"""
  data: Any


class StepError(BaseModel):
  """Tool execution failure exception information for easy serialization"""
  error: str


class SessionState(BaseModel):
  """
  Session immutable snapshot.
  It is a pure data object used to save to a database or deserialize from a database to restore a session.
  """
  session_id: str
  status: SessionStatus

  current_plan: Plan | None = None

  completed_steps: dict[str, StepResult] = Field(default_factory=dict)
  failed_steps: dict[str, StepError] = Field(default_factory=dict)

  start_time: datetime | None = None
  end_time: datetime | None = None
  turn_count: int = 0
  step_count: int = 0


@runtime_checkable
class CheckpointStore(Protocol):
  """
  Persistence protocol.
  """

  # Save/load entire session snapshot
  async def save_session(self, session_id: str, state: SessionState) -> None: ...
  async def load_session(self, session_id: str) -> SessionState | None: ...

  # Granular step updates (avoid rewriting the entire Session every time the Step changes)
  async def save_step_status(self, session_id: str, step_id: str, status: StepStatus) -> None: ...
  async def save_step_result(self, session_id: str, step_id: str, result: Any) -> None: ...
  async def save_step_error(self, session_id: str, step_id: str, error: Exception) -> None: ...