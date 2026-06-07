from __future__ import annotations

from collections.abc import Callable
from typing import Any

from nonoka.core.tool import tool as make_tool
from nonoka.core.types import Capability, RetryPolicy


class ToolRegistry:
  """
  Tool Registry
  Used to register and manage tools
  """
  def __init__(self):
    self._tools: dict[str, Capability] = {}

  def register(
    self,
    func: Callable | None = None,
    *,
    description: str | None = None,
    default_retry: RetryPolicy | None = None,
    default_timeout: float | None = None,
  ):
    """Used to register tools"""
    def wrapper(f: Callable) -> Capability:
      if isinstance(f, Capability) or hasattr(f, "invoke"):
        self.add(f)
        return f 

      t = make_tool(
        f,
        description=description,
        default_retry=default_retry,
        default_timeout=default_timeout,
      )
      
      self.add(t)
      return t

    if func is None:
      return wrapper
    return wrapper(func)

  def add(self, capability: Capability) -> None:
    """Add a capability to the registry"""
    self._tools[capability.name] = capability

  def get_all(self) -> list[Capability]:
    """Get all registered capabilities"""
    return list(self._tools.values())