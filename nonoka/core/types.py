from nonoka.core.event import AgentEvent
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable, TYPE_CHECKING
if TYPE_CHECKING:
  from nonoka.core.agent import Agent

DepsT = TypeVar("DepsT")

@dataclass
class RetryPolicy:
  """Retry Policy"""
  max_retries: int = 3
  backoff: float = 2.0

class RunContext(Generic[DepsT]):
  """Runtime Context, passed to every tool invocation"""
  deps: DepsT
  session_id: str
  agent: "Agent | None"
  memory: Any  # TODO: replace with actual memory type

  def __init__(self, deps: DepsT, session_id: str = "default", agent: "Agent | None" = None, memory: Any = None):
    self.deps = deps
    self.session_id = session_id
    self.agent = agent
    self.memory = memory

  # TODO: replace with actual methods
  async def call_tool(self, name: str, **args: Any) -> Any:
    """Call another tool in the current session"""
    raise NotImplementedError

  async def checkpoint(self, label: str = "") -> None:
    """Manual checkpoint"""
    raise NotImplementedError

  def emit(self, event: AgentEvent) -> None:
    """Emit observability event (AgentEvent)"""
    raise NotImplementedError

@runtime_checkable
class Capability(Protocol):
  """
  Capability Protocol
	Every tool/agent should implement this protocol
  """
  @property
  def name(self) -> str: ...
  
  @property
  def description(self) -> str: ...
  
  @property
  def parameters(self) -> dict[str, Any]: ...
  
  @property
  def returns(self) -> dict[str, Any]: ...
  
  async def invoke(self, ctx: RunContext, arguments: dict[str, Any]) -> Any: ...
  
  def to_json_schema(self) -> dict[str, Any]:
    return {
      "type": "function",
      "function": {
        "name": self.name,
        "description": self.description,
        "parameters": self.parameters,
      },
    }