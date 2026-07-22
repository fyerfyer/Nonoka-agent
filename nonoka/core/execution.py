"""Capability execution metadata and deterministic per-turn scheduling."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable


@dataclass(frozen=True)
class ToolExecution:
  """Declare the side-effect semantics of a capability.

  A capability must opt in to parallel execution with ``read_only=True``.
  Missing metadata is deliberately treated as stateful and serial; this is a
  safe compatibility default for third-party tools whose effects are unknown.
  """

  read_only: bool = False
  mutates_workspace: bool = False
  exclusive: bool = False
  stateful_action: bool = False
  pagination: bool = False

  @property
  def parallel_safe(self) -> bool:
    return (
      self.read_only
      and not self.mutates_workspace
      and not self.exclusive
      and not self.stateful_action
    )

  @property
  def is_stateful(self) -> bool:
    return not self.parallel_safe


UNKNOWN_EXECUTION = ToolExecution(stateful_action=True)


def execution_for(capability: Any | None) -> ToolExecution:
  """Return declared execution metadata, falling back to safe serialization."""
  execution = getattr(capability, "execution", None)
  return execution if isinstance(execution, ToolExecution) else UNKNOWN_EXECUTION


class ToolExecutionCoordinator:
  """Execute a model turn in deterministic, conflict-free waves.

  Consecutive explicitly read-only calls share a bounded concurrent wave.
  Every other call forms a singleton wave, preserving model order for writes,
  terminal actions and capabilities with unknown semantics.
  """

  def __init__(self, max_concurrency: int) -> None:
    self.max_concurrency = max(1, max_concurrency)

  async def execute(
    self,
    calls: list[dict[str, Any]],
    capability_for: Callable[[dict[str, Any]], Any | None],
    invoke: Callable[[dict[str, Any]], Awaitable[Any]],
  ) -> list[Any]:
    results: list[Any] = [None] * len(calls)
    index = 0
    while index < len(calls):
      capability = capability_for(calls[index])
      if not execution_for(capability).parallel_safe:
        try:
          results[index] = await invoke(calls[index])
        except BaseException as exc:  # gather() historically returned errors
          results[index] = exc
        index += 1
        continue

      end = index + 1
      while end < len(calls) and execution_for(capability_for(calls[end])).parallel_safe:
        end += 1

      sem = asyncio.Semaphore(self.max_concurrency)

      async def run_one(call_index: int) -> tuple[int, Any]:
        async with sem:
          try:
            return call_index, await invoke(calls[call_index])
          except BaseException as exc:
            return call_index, exc

      for call_index, value in await asyncio.gather(*[
        run_one(i) for i in range(index, end)
      ]):
        results[call_index] = value
      index = end
    return results
