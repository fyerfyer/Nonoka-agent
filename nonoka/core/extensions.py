"""Constrained, composable enhancements for the ReAct lifecycle.

Extensions may add feedback, validate a proposed final answer, or observe a
tool batch.  They deliberately cannot execute tools, alter tool calls, relax
budgets, or change execution metadata: those remain core safety invariants.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class LoopExtensionContext:
  """Read-only runtime context made available to a loop extension."""

  session: Any
  runner: Any
  prompt: str
  turn: int
  content: str | None = None
  tool_calls: list[dict[str, Any]] = field(default_factory=list)
  tool_results: list[Any] = field(default_factory=list)


@dataclass(frozen=True)
class ExtensionDecision:
  """A bounded extension decision consumed by the core loop.

  ``continue_loop`` injects ``feedback`` as a SYSTEM message and consumes the
  next normal turn. ``failure`` terminates the run.  Neither decision can
  modify tool calls or the execution coordinator's safety policy.
  """

  feedback: str | None = None
  continue_loop: bool = False
  replacement_content: str | None = None
  failure: str | None = None
  details: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class LoopExtension(Protocol):
  """Optional hooks for a normal, non-streaming ReAct execution."""

  name: str

  async def before_turn(self, context: LoopExtensionContext) -> ExtensionDecision | None:
    ...

  async def after_tool_batch(self, context: LoopExtensionContext) -> ExtensionDecision | None:
    ...

  async def before_final_answer(self, context: LoopExtensionContext) -> ExtensionDecision | None:
    ...

  async def after_run(self, context: LoopExtensionContext, result: Any) -> None:
    ...


class LoopExtensionManager:
  """Runs extensions in registration order and records every decision."""

  def __init__(self, extensions: list[LoopExtension] | None = None) -> None:
    self.extensions = list(extensions or [])
    names = [self._name(extension) for extension in self.extensions]
    if len(names) != len(set(names)):
      raise ValueError("Loop extension names must be unique within an Agent")

  @staticmethod
  def _name(extension: Any) -> str:
    return str(getattr(extension, "name", type(extension).__name__))

  async def before_turn(self, context: LoopExtensionContext) -> ExtensionDecision:
    return await self._run_phase("before_turn", context)

  async def after_tool_batch(self, context: LoopExtensionContext) -> ExtensionDecision:
    return await self._run_phase("after_tool_batch", context)

  async def before_final_answer(self, context: LoopExtensionContext) -> ExtensionDecision:
    return await self._run_phase("before_final_answer", context)

  async def after_run(self, context: LoopExtensionContext, result: Any) -> None:
    for extension in self.extensions:
      callback = getattr(extension, "after_run", None)
      if callback is None:
        continue
      await _maybe_await(callback(context, result))
      self._record(context, extension, "after_run", ExtensionDecision())

  async def _run_phase(self, phase: str, context: LoopExtensionContext) -> ExtensionDecision:
    combined = ExtensionDecision(replacement_content=context.content)
    for extension in self.extensions:
      callback = getattr(extension, phase, None)
      if callback is None:
        continue
      raw = await _maybe_await(callback(context))
      decision = _coerce_decision(raw)
      self._record(context, extension, phase, decision)
      if decision.replacement_content is not None:
        context.content = decision.replacement_content
        combined = ExtensionDecision(
          feedback=combined.feedback,
          continue_loop=combined.continue_loop,
          replacement_content=decision.replacement_content,
          failure=combined.failure,
          details={**combined.details, **decision.details},
        )
      if decision.feedback:
        combined = ExtensionDecision(
          feedback="\n\n".join(part for part in (combined.feedback, decision.feedback) if part),
          continue_loop=combined.continue_loop or decision.continue_loop,
          replacement_content=combined.replacement_content,
          failure=combined.failure,
          details={**combined.details, **decision.details},
        )
      if decision.continue_loop:
        combined = ExtensionDecision(
          feedback=combined.feedback,
          continue_loop=True,
          replacement_content=combined.replacement_content,
          failure=combined.failure,
          details={**combined.details, **decision.details},
        )
      if decision.failure:
        return ExtensionDecision(
          feedback=combined.feedback,
          continue_loop=False,
          replacement_content=combined.replacement_content,
          failure=decision.failure,
          details={**combined.details, **decision.details},
        )
    return combined

  def _record(
    self, context: LoopExtensionContext, extension: Any, phase: str, decision: ExtensionDecision,
  ) -> None:
    trace = getattr(context.session, "trace", None)
    if trace is not None:
      trace.record_extension(
        name=self._name(extension), phase=phase, turn=context.turn,
        decision={
          "continue_loop": decision.continue_loop,
          "replacement_content": decision.replacement_content,
          "failure": decision.failure,
          "details": decision.details,
        },
      )


async def _maybe_await(value: Any) -> Any:
  return await value if inspect.isawaitable(value) else value


def _coerce_decision(value: Any) -> ExtensionDecision:
  if value is None:
    return ExtensionDecision()
  if isinstance(value, ExtensionDecision):
    return value
  raise TypeError(
    "Loop extension hooks must return ExtensionDecision or None, "
    f"got {type(value).__name__}"
  )
