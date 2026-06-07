from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar, runtime_checkable, Generic

if TYPE_CHECKING:
  from nonoka.core import RunContext

DepsT = TypeVar("DepsT")
ResultT = TypeVar("ResultT")

from enum import Enum


class ToolErrorAction(str, Enum):
  RETRY = "retry"
  HALT = "halt"
  REPORT = "report"
  FAIL = "fail"


@dataclass
class RunResult(Generic[ResultT]):
  """Result of an Agent run."""
  success: bool
  data: ResultT | None = None
  session: Any | None = None
  error: str | None = None
  error_type: str | None = None  # "llm_error" | "tool_error" | "timeout" | "cancelled" | ...


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

  async def invoke(self, ctx: RunContext, arguments: dict[str, Any]) -> Any: ...

  def to_json_schema(self) -> dict[str, Any]: ...
