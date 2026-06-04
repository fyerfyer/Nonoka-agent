from typing import Any, Generic, TypeVar, TYPE_CHECKING
from nonoka.core.event import AgentEvent

if TYPE_CHECKING:
  from nonoka.core.agent import Agent
  from nonoka.core.memory import WorkingMemory
  from nonoka.core.session import Session

DepsT = TypeVar("DepsT")


from nonoka.core.logger import get_logger

_logger = get_logger("nonoka")


class RunContext(Generic[DepsT]):
  """Runtime context, passed to every tool invocation.

  ``RunContext`` is a read-only view of the current ``Session``.
  It gives tools access to *deps*, *memory*, and *session_id* without
  exposing mutable execution state (``completed_steps``, ``current_plan``).
  """

  def __init__(self, session: "Session"):
    self._session = session

  # ------------------------------------------------------------------ #
  # Read-only properties — proxied from Session
  # ------------------------------------------------------------------ #

  @property
  def deps(self) -> DepsT:
    return self._session.deps

  @property
  def session_id(self) -> str:
    return self._session.session_id

  @property
  def agent(self) -> "Agent | None":
    return self._session.agent

  @property
  def memory(self) -> "WorkingMemory | None":
    return self._session.memory

  @property
  def session(self) -> "Session":
    """Read-only access to the underlying ``Session``.

    Tools can inspect ``session.completed_steps``, ``session.turn_count``,
    etc., but mutating execution state directly is discouraged.
    """
    return self._session

  # ------------------------------------------------------------------ #
  # Tool helper
  # ------------------------------------------------------------------ #

  async def call_tool(self, name: str, **args: Any) -> Any:
    """Call another tool in the current session."""
    if not self.agent:
      raise RuntimeError("Cannot call tool: RunContext has no bound Agent.")

    target_tool = next((t for t in self.agent.tools if t.name == name), None)
    if not target_tool:
      raise ValueError(f"Tool '{name}' not found in the current Agent.")

    return await target_tool.invoke(self, args)

  def emit(self, event: AgentEvent) -> None:
    """Emit a structured observability event via structlog."""
    event_type = event.type.value if hasattr(event.type, "value") else str(event.type)
    _logger.info(
      f"[{event_type}] session={event.session_id}",
      agent_event=event.model_dump(mode="json"),
    )

  async def checkpoint(self, label: str = "") -> None:
    """Log a checkpoint request (actual save is handled by the scheduler)."""
    self.emit(AgentEvent(
      type="checkpoint.requested",
      session_id=self.session_id,
      data={"label": label}
    ))
