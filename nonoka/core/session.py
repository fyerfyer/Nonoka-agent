from enum import Enum
from typing import Any, TYPE_CHECKING
from pydantic import BaseModel, Field
from datetime import datetime
from nonoka.core.plan import Plan

if TYPE_CHECKING:
  from nonoka.core.agent import Agent
  from nonoka.core.memory import WorkingMemory


class SessionStatus(str, Enum):
  """Session lifecycle status."""
  CREATED = "created"
  RUNNING = "running"
  PAUSED = "paused"
  COMPLETED = "completed"
  FAILED = "failed"


class StepStatus(str, Enum):
  """Step lifecycle status."""
  PENDING = "pending"
  RUNNING = "running"
  COMPLETED = "completed"
  FAILED = "failed"


class StepResult(BaseModel):
  """Serialisable record of a successfully executed step."""
  data: Any


class StepFailure(BaseModel):
  """Serialisable record of a failed step execution.

  This is a data-transfer object (not an Exception) used for checkpointing
  and state reconstruction.
  """
  error_type: str
  message: str
  traceback: str | None = None


class SessionState(BaseModel):
  """
  Immutable snapshot of a Session.

  This is a pure data object used to save to a database or deserialise
  from a database to restore a session.
  """
  session_id: str
  status: SessionStatus

  current_plan: Plan | None = None

  completed_steps: dict[str, StepResult] = Field(default_factory=dict)
  failed_steps: dict[str, StepFailure] = Field(default_factory=dict)
  step_statuses: dict[str, StepStatus] = Field(default_factory=dict)

  start_time: datetime | None = None
  end_time: datetime | None = None
  turn_count: int = 0
  step_count: int = 0


class Session:
  """Mutable runtime state for a single agent execution."""

  def __init__(
    self,
    session_id: str,
    agent: "Agent",
    deps: Any = None,
    memory: "WorkingMemory | None" = None,
  ):
    self.session_id = session_id
    self.agent = agent
    self.deps = deps
    self.memory = memory

    self.status = SessionStatus.CREATED
    self.current_plan: Plan | None = None
    self.completed_steps: dict[str, StepResult] = {}
    self.failed_steps: dict[str, StepFailure] = {}
    self.step_statuses: dict[str, StepStatus] = {}

    self.start_time = datetime.now()
    self.end_time: datetime | None = None
    self.turn_count = 0
    self.step_count = 0

  def to_state(self) -> SessionState:
    """Serialize to immutable state for checkpoint."""
    return SessionState(
      session_id=self.session_id,
      status=self.status,
      current_plan=self.current_plan,
      completed_steps=self.completed_steps.copy(),
      failed_steps=self.failed_steps.copy(),
      step_statuses=self.step_statuses.copy(),
      start_time=self.start_time,
      end_time=self.end_time,
      turn_count=self.turn_count,
      step_count=self.step_count,
    )

  @classmethod
  def from_state(
    cls,
    state: SessionState,
    agent: "Agent",
    deps: Any = None,
    memory: "WorkingMemory | None" = None,
  ) -> "Session":
    """Restore from checkpoint."""
    session = cls(
      session_id=state.session_id,
      agent=agent,
      deps=deps,
      memory=memory,
    )
    session.status = state.status
    session.current_plan = state.current_plan
    session.completed_steps = state.completed_steps.copy()
    session.failed_steps = state.failed_steps.copy()
    session.step_statuses = state.step_statuses.copy()
    session.start_time = state.start_time
    session.end_time = state.end_time
    session.turn_count = state.turn_count
    session.step_count = state.step_count
    return session
