import asyncio
from typing import Any, Generic, TypeVar, TYPE_CHECKING
from nonoka.core.event import AgentEvent
from nonoka.core.memory import WorkingMemory

if TYPE_CHECKING:
  from nonoka.core.agent import Agent

DepsT = TypeVar("DepsT")


class RunContext(Generic[DepsT]):
  """Runtime Context, passed to every tool invocation"""
  deps: DepsT
  session_id: str
  agent: "Agent | None"
  memory: WorkingMemory | None 

  def __init__(
    self,
    deps: DepsT,
    session_id: str = "default",
    agent: "Agent | None" = None,
    memory: WorkingMemory | None = None
  ):
    self.deps = deps
    self.session_id = session_id
    self.agent = agent
    self.memory = memory

  async def call_tool(self, name: str, **args: Any) -> Any:
    """Call another tool in the current session"""
    if not self.agent:
      raise RuntimeError("Cannot call tool: RunContext has no bound Agent.")

    target_tool = next((t for t in self.agent.tools if t.name == name), None)
    if not target_tool:
      raise ValueError(f"Tool '{name}' not found in the current Agent.")

    return await target_tool.invoke(self, args)

  async def checkpoint(self, label: str = "") -> None:
    """Trigget a checkpoint manually"""
    self.emit(AgentEvent(
      type="checkpoint.requested",
      session_id=self.session_id,
      data={"label": label}
    ))

  def emit(self, event: AgentEvent) -> None:
    """Emit observability event (AgentEvent)"""
    if not self.agent:
      return

    backends = getattr(self.agent, 'observability_backends', [])
    for backend in backends:
      if hasattr(backend, "on_event"):
        asyncio.create_task(backend.on_event(event))