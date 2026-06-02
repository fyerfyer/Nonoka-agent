from typing import TYPE_CHECKING
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
  from nonoka.core import RunContext

DepsT = TypeVar("DepsT")

@dataclass
class RetryPolicy:
  """Retry Policy"""
  max_retries: int = 3
  backoff: float = 2.0

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