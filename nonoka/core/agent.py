from nonoka.core.checkpoint import CheckpointStore
from dataclasses import dataclass, field
from typing import Generic, TypeVar, Any
from collections.abc import Callable

from nonoka.core.types import Capability, RetryPolicy
from nonoka.core.tool import tool as make_tool
from nonoka.core.registry import ToolRegistry

DepsT = TypeVar("DepsT")
ResultT = TypeVar("ResultT")


@dataclass(frozen=True)
class Agent(Generic[DepsT, ResultT]):
  """
  Agent is a state-less configuration object that holds the model, tools, and skills.
  Runtime state should be stored in Session.
  """
  model: str
  tools: list[Capability] = field(default_factory=list)
  skills: list[Capability] = field(default_factory=list)  # TODO: implement Skill Protocol
  system_prompt: str = ""

  # Generic type hints for runtime type inference
  deps_type: type[DepsT] | None = None
  result_type: type[ResultT] | None = None

  # Default execution policy
  max_turns: int = 10
  max_steps: int = 50
  default_retry: RetryPolicy = field(default_factory=RetryPolicy)
  default_timeout: float | None = None

  # TODO: use protocol to replace Any
  memory: Any | None = None
  checkpoint_store: CheckpointStore | None = None
  scheduler: Any | None = None

  def tool(
    self,
    func: Callable | None = None,
    *,
    description: str | None = None,
    default_retry: RetryPolicy | None = None,
    default_timeout: float | None = None,
  ):
    """
    Pythonic way to create a tool with decorator
    Usage:
      @agent.tool()
      def tool_func():
        pass
    """
    def wrapper(f: Callable) -> Capability:
      if isinstance(f, Capability) or hasattr(f, "invoke"):
        self.tools.append(f)
        return f 
        
      t = make_tool(
        f,
        description=description,
        default_retry=default_retry,
        default_timeout=default_timeout,
      )
      self.tools.append(t)
      return t

    if func is None:
      return wrapper
    return wrapper(func)

  def add_tools(self, registry: ToolRegistry) -> None:
    """
    Explicitly bind tools from a ToolRegistry
    """
    self.tools.extend(registry.get_all())