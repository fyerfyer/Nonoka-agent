from __future__ import annotations

from collections.abc import Callable, Iterator
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

  def remove(self, name: str) -> Capability | None:
    """Remove a capability by name and return it (or None if not found)."""
    return self._tools.pop(name, None)

  def get(self, name: str) -> Capability | None:
    """Get a capability by name."""
    return self._tools.get(name)

  def has(self, name: str) -> bool:
    """Check whether a capability is registered."""
    return name in self._tools

  def names(self) -> list[str]:
    """Return a list of all registered capability names."""
    return list(self._tools.keys())

  def get_all(self) -> list[Capability]:
    """Get all registered capabilities"""
    return list(self._tools.values())

  def clear(self) -> None:
    """Remove all capabilities."""
    self._tools.clear()

  # -- Collection-like interface ----------------------------------------

  def __contains__(self, name: str) -> bool:
    return name in self._tools

  def __iter__(self) -> Iterator[Capability]:
    return iter(self._tools.values())

  def __len__(self) -> int:
    return len(self._tools)

  def __repr__(self) -> str:
    return f"<ToolRegistry tools={self.names()}>"