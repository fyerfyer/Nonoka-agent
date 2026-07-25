from __future__ import annotations

import asyncio
import copy
from enum import Enum
from typing import Any, TYPE_CHECKING
from pydantic import BaseModel, Field
from datetime import datetime
from nonoka.core.plan import Plan
from nonoka.core.runtime import (
  CompletionContract,
  RuntimeLimits,
  SessionRuntimeState,
  TerminalReason,
  Termination,
)

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
  CANCELLED = "cancelled"


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

  # Memory snapshot for conversational checkpoint/resume
  memory_entries: list[dict[str, Any]] = Field(default_factory=list)

  start_time: datetime | None = None
  end_time: datetime | None = None
  turn_count: int = 0
  step_count: int = 0
  trace: dict[str, Any] | None = None
  runtime_state: SessionRuntimeState | None = None
  completion_contract: CompletionContract | None = None
  # Namespaced, JSON-serialisable state owned by loop extensions. Keeping it
  # in the checkpoint prevents external-tool pause/resume boundaries from
  # resetting progress detectors and bounded repair counters.
  extension_state: dict[str, dict[str, Any]] = Field(default_factory=dict)


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

    configured = agent.runtime_limits or RuntimeLimits()
    effective_limits = RuntimeLimits(
      max_model_turns=(
        configured.max_model_turns
        if configured.max_model_turns is not None
        else agent.max_turns
      ),
      max_tool_calls=(
        configured.max_tool_calls
        if configured.max_tool_calls is not None
        else agent.max_steps
      ),
      wall_timeout_seconds=configured.wall_timeout_seconds,
      model_timeout_seconds=configured.model_timeout_seconds or agent.default_timeout,
      max_context_bytes=configured.max_context_bytes,
      max_context_tokens=configured.max_context_tokens,
      max_tool_messages=configured.max_tool_messages,
      max_external_result_bytes=configured.max_external_result_bytes,
    )
    self.runtime_state = SessionRuntimeState.create(effective_limits)
    self.completion_contract = agent.completion_contract
    self.extension_state: dict[str, dict[str, Any]] = {}

    # Cancellation support — asyncio.Event allows cooperative cancellation
    # and is safe to check across async boundaries.
    self._cancel_event = asyncio.Event()
    from nonoka.core.trace import ExecutionTrace
    self.trace = ExecutionTrace()
    self.trace.record_generation(
      model=agent.model, temperature=agent.temperature, max_tokens=agent.max_tokens,
    )

  # ------------------------------------------------------------------ #
  # Cancellation API
  # ------------------------------------------------------------------ #

  def cancel(self) -> None:
    """Request cancellation of this session.

    Cancellation is cooperative — paradigms check ``is_cancelled`` at
    safe boundaries (between turns / layers) and raise ``CancelledError``.
    """
    self._cancel_event.set()
    now = datetime.now().astimezone()
    self.runtime_state.cancelled_at = now
    self.runtime_state.termination = Termination(
      reason=TerminalReason.CANCELLED,
      message=f"Session {self.session_id} was cancelled by external request.",
    )

  @property
  def is_cancelled(self) -> bool:
    """Whether cancellation has been requested."""
    return self._cancel_event.is_set()

  def check_cancelled(self) -> None:
    """Raise ``CancelledError`` if cancellation has been requested."""
    if self._cancel_event.is_set():
      from nonoka.core.errors import CancelledError
      raise CancelledError(
        f"Session {self.session_id} was cancelled by external request."
      )

  def terminate(self, termination: Termination) -> None:
    """Persist a typed terminal reason on the mutable session."""
    self.runtime_state.termination = termination

  def check_runtime(self) -> None:
    """Raise when cancellation, deadline, or a previously terminal state applies."""
    from nonoka.core.errors import RuntimeTerminatedError

    if self.runtime_state.termination is not None:
      raise RuntimeTerminatedError(self.runtime_state.termination)
    remaining = self.runtime_state.remaining_seconds()
    if remaining is not None and remaining <= 0:
      termination = Termination(
        reason=TerminalReason.DEADLINE_EXCEEDED,
        message=f"Session {self.session_id} exceeded its wall-clock deadline.",
        dimension="wall_timeout_seconds",
        limit=self.runtime_state.limits.wall_timeout_seconds,
        used=self.runtime_state.limits.wall_timeout_seconds,
      )
      self.terminate(termination)
      raise RuntimeTerminatedError(termination)
    self.check_cancelled()

  def begin_model_turn(self) -> int:
    """Reserve one cumulative model turn and return its one-based index."""
    from nonoka.core.errors import RuntimeTerminatedError

    self.check_runtime()
    limits = self.runtime_state.limits
    usage = self.runtime_state.usage
    if limits.max_model_turns is not None and usage.model_turns >= limits.max_model_turns:
      termination = Termination(
        reason=TerminalReason.TURN_BUDGET_EXHAUSTED,
        message=f"Max turns ({limits.max_model_turns}) exceeded for session {self.session_id}",
        dimension="max_model_turns",
        limit=limits.max_model_turns,
        used=usage.model_turns,
      )
      self.terminate(termination)
      raise RuntimeTerminatedError(termination)
    usage.model_turns += 1
    self.turn_count = usage.model_turns
    return usage.model_turns

  def reserve_tool_calls(self, count: int, *, external_count: int = 0, last_tool: str | None = None) -> None:
    """Reserve a model-selected tool batch before local or host execution."""
    from nonoka.core.errors import RuntimeTerminatedError

    self.check_runtime()
    limits = self.runtime_state.limits
    usage = self.runtime_state.usage
    if limits.max_tool_calls is not None and usage.tool_calls + count > limits.max_tool_calls:
      termination = Termination(
        reason=TerminalReason.TOOL_BUDGET_EXHAUSTED,
        message=f"Session {self.session_id} exhausted its tool-call budget.",
        dimension="max_tool_calls",
        limit=limits.max_tool_calls,
        used=usage.tool_calls,
        diagnostics={"requested": count, "last_tool": last_tool},
      )
      self.terminate(termination)
      raise RuntimeTerminatedError(termination)
    usage.tool_calls += count
    usage.external_tool_calls += external_count
    usage.last_tool = last_tool

  async def enforce_context_budget(self) -> None:
    """Apply the configured context compactor and persist its measurements."""
    if self.memory is None:
      return
    from nonoka.core.errors import ContextBudgetExceeded, RuntimeTerminatedError

    try:
      metrics = await self.memory.enforce_budget(self.runtime_state.limits)
    except ContextBudgetExceeded as exc:
      termination = Termination(
        reason=TerminalReason.CONTEXT_BUDGET_EXHAUSTED,
        message=str(exc),
        dimension="context",
        diagnostics={**exc.metrics, **exc.limits},
      )
      self.terminate(termination)
      raise RuntimeTerminatedError(termination) from exc
    if metrics is not None:
      usage = self.runtime_state.usage
      usage.context_bytes = metrics.serialized_bytes
      usage.context_tokens = metrics.tokens
      usage.tool_messages = metrics.tool_messages

  def completion_feedback(self) -> str | None:
    """Return corrective feedback, or raise after the contract budget is spent."""
    contract = self.completion_contract
    if contract is None:
      return None
    usage = self.runtime_state.usage
    unmet = contract.unmet_requirements(usage)
    if not unmet:
      return None
    if usage.correction_count < contract.max_corrections:
      usage.correction_count += 1
      if contract.observation_review_unmet(usage):
        # A completion correction creates a new evidence epoch: observations
        # made before the feedback cannot satisfy the requested re-check.
        usage.observation_feedback_after = usage.observation_count
      return (
        "[Completion contract] The task is not complete yet. "
        + "; ".join(unmet)
        + ". Continue working, verify the result, then provide the final answer."
      )

    # An external evaluator is sometimes the authoritative scorer for a
    # workspace.  In that mode a contract is still valuable: it gives the
    # model bounded corrective feedback and persists the unmet evidence, but
    # it must not discard a concrete workspace state before the scorer can
    # inspect it.  Strict remains the default for callers that need the
    # contract itself to be a hard success criterion.
    if contract.enforcement == "advisory":
      usage.completion_warning_count += 1
      usage.latest_completion_warning = "; ".join(unmet)
      return None

    from nonoka.core.errors import RuntimeTerminatedError
    termination = Termination(
      reason=TerminalReason.COMPLETION_CONTRACT_UNMET,
      message="The completion contract remained unsatisfied after the allowed correction.",
      dimension="completion_contract",
      used=usage.correction_count,
      limit=contract.max_corrections,
      diagnostics={"unmet": unmet},
    )
    self.terminate(termination)
    raise RuntimeTerminatedError(termination)

  # ------------------------------------------------------------------ #
  # Serialization
  # ------------------------------------------------------------------ #

  def to_state(self) -> SessionState:
    """Serialize to immutable state for checkpoint."""
    memory_entries: list[dict[str, Any]] = []
    if self.memory is not None:
      memory_entries = [entry.model_dump(mode="json") for entry in self.memory.entries]

    return SessionState(
      session_id=self.session_id,
      status=self.status,
      current_plan=self.current_plan,
      completed_steps=self.completed_steps.copy(),
      failed_steps=self.failed_steps.copy(),
      step_statuses=self.step_statuses.copy(),
      memory_entries=memory_entries,
      start_time=self.start_time,
      end_time=self.end_time,
      turn_count=self.turn_count,
      step_count=self.step_count,
      trace=self.trace.to_dict(),
      runtime_state=self.runtime_state,
      completion_contract=self.completion_contract,
      extension_state=copy.deepcopy(self.extension_state),
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
    if state.runtime_state is not None:
      session.runtime_state = state.runtime_state
    session.completion_contract = state.completion_contract
    session.extension_state = copy.deepcopy(state.extension_state)
    if session.runtime_state.is_cancelled:
      session._cancel_event.set()
    if state.trace is not None:
      from nonoka.core.trace import ExecutionTrace
      session.trace = ExecutionTrace.from_dict(state.trace)

    # Restore memory entries if memory is provided and state has a snapshot
    if memory is not None and state.memory_entries:
      from nonoka.core.memory import MemoryEntry
      memory.entries = [MemoryEntry(**entry) for entry in state.memory_entries]

    return session
