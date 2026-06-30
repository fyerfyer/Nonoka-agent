"""
Lightweight Hooks / Middleware mechanism.

Provides the ability to inject custom logic at key points of the Agent
execution lifecycle.  Supports three registration styles:

1. Subclass ``Hooks`` and override methods.
2. Pass a list of callables to the ``Hooks`` constructor.
3. Register via decorators ``@hooks.on_xxx``.

All hook points are async; sync functions are automatically wrapped.

Usage::

    # Style 1: subclass
    class LoggingHooks(Hooks):
        async def on_llm_request(self, ctx, messages, tools):
            print(f"LLM call with {len(messages)} messages")

    runner = Runner(hooks=LoggingHooks())

    # Style 2: list of functions
    async def log_request(ctx, messages, tools):
        print(f"LLM call with {len(messages)} messages")

    runner = Runner(hooks=Hooks(on_llm_request=[log_request]))

    # Style 3: decorator
    hooks = Hooks()

    @hooks.on_llm_request
    async def log_request(ctx, messages, tools):
        print(f"LLM call with {len(messages)} messages")

    runner = Runner(hooks=hooks)
"""

from __future__ import annotations

import inspect
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

if __debug__:
  from nonoka.core.session import Session
  from nonoka.core.agent import Agent
  from nonoka.core.llm import LLMMessage, LLMResponse
  from nonoka.core.types import RunResult


# --------------------------------------------------------------------------- #
# Hook signatures (for documentation and type-checking)
# --------------------------------------------------------------------------- #

SessionStartHook = Callable[["HookContext"], Awaitable[None] | None]
SessionEndHook = Callable[["HookContext", "RunResult"], Awaitable[None] | None]
LLMRequestHook = Callable[["HookContext", list["LLMMessage"], list[dict[str, Any]] | None], Awaitable[None] | None]
LLMResponseHook = Callable[["HookContext", "LLMResponse"], Awaitable[None] | None]
ToolStartHook = Callable[["HookContext", str, dict[str, Any]], Awaitable[None] | None]
ToolStartInterceptHook = Callable[["HookContext", str, dict[str, Any]], Awaitable[dict[str, Any]] | dict[str, Any]]
ToolEndHook = Callable[["HookContext", str, dict[str, Any], Any, Exception | None], Awaitable[None] | None]
PlanStartHook = Callable[["HookContext"], Awaitable[None] | None]
PlanStepStartHook = Callable[["HookContext", str, str, dict[str, Any]], Awaitable[None] | None]
PlanStepEndHook = Callable[["HookContext", str, str, Any, Exception | None], Awaitable[None] | None]


# --------------------------------------------------------------------------- #
# HookContext — passed to every hook invocation
# --------------------------------------------------------------------------- #

class HookContext:
  """Runtime context passed to every hook invocation.

  Attributes:
    session: The current execution session (contains agent, deps, memory).
    runner: The current Runner instance (used to access checkpoint_store, etc.).
    timestamp: The hook trigger time (UTC).
    extra: An extra data dict that can be shared between hooks.
  """

  def __init__(
    self,
    session: "Session",
    runner: Any,
    timestamp: datetime | None = None,
    extra: dict[str, Any] | None = None,
  ):
    self.session = session
    self.runner = runner
    self.timestamp = timestamp or datetime.now(timezone.utc)
    self.extra = extra or {}

  @property
  def agent(self) -> "Agent":
    """Current Agent configuration (proxied from session)."""
    return self.session.agent


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

def _normalize_hook(fn: Callable[..., Any]) -> Callable[..., Awaitable[None]]:
  """Normalize a sync or async function into an async function."""
  if inspect.iscoroutinefunction(fn):
    return fn  # type: ignore[return-value]

  async def _wrapper(*args: Any, **kwargs: Any) -> Any:
    return fn(*args, **kwargs)

  return _wrapper


async def _run_hooks(hooks: list[Callable[..., Awaitable[None]]], *args: Any, **kwargs: Any) -> None:
  """Execute a list of hooks sequentially (exceptions propagate naturally)."""
  for hook in hooks:
    await hook(*args, **kwargs)


# --------------------------------------------------------------------------- #
# Hooks — user-facing API
# --------------------------------------------------------------------------- #

class Hooks:
  """Lightweight hooks / middleware registry.

  Each hook point can be:
  * ``None`` — no listener at this hook point.
  * A single callable — one listener.
  * A list of callables — multiple listeners executed in order.

  Subclasses can register hooks by overriding methods with the same name;
  constructor arguments take priority over subclass methods (explicit override).
  """

  _HOOK_POINTS: tuple[str, ...] = (
    "on_session_start",
    "on_session_end",
    "on_llm_request",
    "on_llm_response",
    "on_tool_start",
    "on_tool_start_intercept",
    "on_tool_end",
    "on_plan_start",
    "on_plan_step_start",
    "on_plan_step_end",
  )

  def __init__(
    self,
    on_session_start: SessionStartHook | list[SessionStartHook] | None = None,
    on_session_end: SessionEndHook | list[SessionEndHook] | None = None,
    on_llm_request: LLMRequestHook | list[LLMRequestHook] | None = None,
    on_llm_response: LLMResponseHook | list[LLMResponseHook] | None = None,
    on_tool_start: ToolStartHook | list[ToolStartHook] | None = None,
    on_tool_start_intercept: ToolStartInterceptHook | list[ToolStartInterceptHook] | None = None,
    on_tool_end: ToolEndHook | list[ToolEndHook] | None = None,
    on_plan_start: PlanStartHook | list[PlanStartHook] | None = None,
    on_plan_step_start: PlanStepStartHook | list[PlanStepStartHook] | None = None,
    on_plan_step_end: PlanStepEndHook | list[PlanStepEndHook] | None = None,
  ):
    # Internal storage: dict of hook point -> list of normalized callables
    self._store: dict[str, list[Callable[..., Awaitable[None]]]] = {}

    # Process constructor arguments
    for point in self._HOOK_POINTS:
      raw = locals()[point]
      self._store[point] = self._normalize_list(raw)

    # Subclass method discovery: if a subclass defined e.g. ``async def on_session_start(self, ctx): ...``
    # and the user did NOT pass an explicit constructor arg for it, use the method.
    for point in self._HOOK_POINTS:
      if not self._store[point]:
        method = getattr(type(self), point, None)
        if method is not None and method is not getattr(Hooks, point, None):
          self._store[point] = [_normalize_hook(method.__get__(self, type(self)))]

  # ------------------------------------------------------------------ #
  # Normalization helper
  # ------------------------------------------------------------------ #

  @staticmethod
  def _normalize_list(raw: Any) -> list[Callable[..., Awaitable[None]]]:
    """Normalize a raw hook value (None / callable / list) into a list."""
    if raw is None:
      return []
    if isinstance(raw, list):
      return [_normalize_hook(fn) for fn in raw]
    return [_normalize_hook(raw)]

  # ------------------------------------------------------------------ #
  # Public: decorator registration API
  # ------------------------------------------------------------------ #

  def _register(self, attr: str, fn: Callable[..., Any]) -> Callable[..., Any]:
    """Register a function to the specified hook point (used internally by decorators)."""
    self._store.setdefault(attr, [])
    self._store[attr].append(_normalize_hook(fn))
    return fn

  def on_session_start(self, fn: SessionStartHook) -> SessionStartHook:
    """Decorator: register a session-start hook."""
    return self._register("on_session_start", fn)  # type: ignore[return-value]

  def on_session_end(self, fn: SessionEndHook) -> SessionEndHook:
    """Decorator: register a session-end hook."""
    return self._register("on_session_end", fn)  # type: ignore[return-value]

  def on_llm_request(self, fn: LLMRequestHook) -> LLMRequestHook:
    """Decorator: register an LLM-request hook."""
    return self._register("on_llm_request", fn)  # type: ignore[return-value]

  def on_llm_response(self, fn: LLMResponseHook) -> LLMResponseHook:
    """Decorator: register an LLM-response hook."""
    return self._register("on_llm_response", fn)  # type: ignore[return-value]

  def on_tool_start(self, fn: ToolStartHook) -> ToolStartHook:
    """Decorator: register a tool-start hook."""
    return self._register("on_tool_start", fn)  # type: ignore[return-value]

  def on_tool_start_intercept(self, fn: ToolStartInterceptHook) -> ToolStartInterceptHook:
    """Decorator: register a tool-start intercept hook.

    Intercept hooks can inspect and *modify* the arguments dict that will
    be passed to the tool.  They are executed in order; each receives the
    arguments returned by the previous hook.  The final dict is what the
    tool actually receives.
    """
    self._store.setdefault("on_tool_start_intercept", [])
    self._store["on_tool_start_intercept"].append(_normalize_hook(fn))  # type: ignore[arg-type]
    return fn  # type: ignore[return-value]

  def on_tool_end(self, fn: ToolEndHook) -> ToolEndHook:
    """Decorator: register a tool-end hook."""
    return self._register("on_tool_end", fn)  # type: ignore[return-value]

  def on_plan_start(self, fn: PlanStartHook) -> PlanStartHook:
    """Decorator: register a plan-start hook."""
    return self._register("on_plan_start", fn)  # type: ignore[return-value]

  def on_plan_step_start(self, fn: PlanStepStartHook) -> PlanStepStartHook:
    """Decorator: register a plan-step-start hook."""
    return self._register("on_plan_step_start", fn)  # type: ignore[return-value]

  def on_plan_step_end(self, fn: PlanStepEndHook) -> PlanStepEndHook:
    """Decorator: register a plan-step-end hook."""
    return self._register("on_plan_step_end", fn)  # type: ignore[return-value]

  # ------------------------------------------------------------------ #
  # Public: emit helpers (used by Runner / paradigms)
  # ------------------------------------------------------------------ #

  async def emit_session_start(self, ctx: HookContext) -> None:
    await _run_hooks(self._store.get("on_session_start", []), ctx)

  async def emit_session_end(self, ctx: HookContext, result: "RunResult") -> None:
    await _run_hooks(self._store.get("on_session_end", []), ctx, result)

  async def emit_llm_request(self, ctx: HookContext, messages: list["LLMMessage"], tools: list[dict[str, Any]] | None) -> None:
    await _run_hooks(self._store.get("on_llm_request", []), ctx, messages, tools)

  async def emit_llm_response(self, ctx: HookContext, response: "LLMResponse") -> None:
    await _run_hooks(self._store.get("on_llm_response", []), ctx, response)

  async def emit_tool_start(self, ctx: HookContext, tool_name: str, arguments: dict[str, Any]) -> None:
    await _run_hooks(self._store.get("on_tool_start", []), ctx, tool_name, arguments)

  async def emit_tool_start_intercept(self, ctx: HookContext, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Run intercept hooks sequentially, passing the return value through.

    Each hook receives ``(ctx, tool_name, arguments)`` and must return a
    dict (possibly modified).  The final dict is returned to the caller.

    If no intercept hooks are registered, ``arguments`` is returned unchanged.
    """
    effective = arguments
    for hook in self._store.get("on_tool_start_intercept", []):
      effective = await hook(ctx, tool_name, effective)  # type: ignore[assignment,operator]
      if not isinstance(effective, dict):
        raise TypeError(
          f"Intercept hook for '{tool_name}' must return a dict, got {type(effective).__name__}"
        )
    return effective

  async def emit_tool_end(
    self,
    ctx: HookContext,
    tool_name: str,
    arguments: dict[str, Any],
    result: Any,
    error: Exception | None,
  ) -> None:
    await _run_hooks(self._store.get("on_tool_end", []), ctx, tool_name, arguments, result, error)

  async def emit_plan_start(self, ctx: HookContext) -> None:
    await _run_hooks(self._store.get("on_plan_start", []), ctx)

  async def emit_plan_step_start(self, ctx: HookContext, step_id: str, tool_name: str, arguments: dict[str, Any]) -> None:
    await _run_hooks(self._store.get("on_plan_step_start", []), ctx, step_id, tool_name, arguments)

  async def emit_plan_step_end(
    self,
    ctx: HookContext,
    step_id: str,
    tool_name: str,
    result: Any,
    error: Exception | None,
  ) -> None:
    await _run_hooks(self._store.get("on_plan_step_end", []), ctx, step_id, tool_name, result, error)
