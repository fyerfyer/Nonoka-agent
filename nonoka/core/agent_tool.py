from __future__ import annotations

import asyncio
import hashlib
from contextlib import suppress
from enum import Enum
from typing import Any, Callable

from nonoka.core.agent import Agent
from nonoka.core.context import RunContext
from nonoka.core.types import Capability, RunResult
from nonoka.core.logger import get_logger

_logger = get_logger("nonoka.agent_tool")


class MemoryStrategy(str, Enum):
  """Strategy for how a sub-agent inherits (or shares) parent memory."""

  ISOLATE = "isolate"
  """Child session starts with completely empty memory."""

  INHERIT = "inherit"
  """Child session copies the last *N* memory entries from the parent."""

  SHARE = "share"
  """Child session uses the same ``WorkingMemory`` object as the parent.

  .. warning::
    Both parent and child agents will read and write the same memory.
    This is powerful for tight collaboration but can cause interference.
  """


class AgentTool(Capability):
  """Wrap an ``Agent`` as a ``Capability`` so it can be called as a tool.

  This is the minimal viable Multi-Agent pattern: one Agent invokes another
  through the standard ReAct tool-calling loop.  There is no orchestrator
  and no message bus — just "Agent calls Agent".

  Usage::

    reviewer = Agent(
      model="claude-sonnet-4-6",
      system_prompt="You are a security reviewer...",
      tools=[read_file, check_security],
    )

    main = Agent(
      model="gpt-4o",
      tools=[
        search_web,
        AgentTool(
          agent=reviewer,
          name="security_reviewer",
          description="When code security review is needed...",
          memory_strategy=MemoryStrategy.ISOLATE,
        ),
      ],
    )

    runner = Runner()
    result = await runner.run_react(main, "Review this project for security issues.")

  Args:
    agent: The sub-agent configuration to execute.
    name: Tool name exposed to the LLM.  Defaults to ``agent_{model}``.
    description: Tool description exposed to the LLM.
    memory_strategy: How child memory relates to parent memory.
    max_depth: Maximum nesting depth (default 3).  Prevents runaway recursion.
    result_extractor: Callable ``(RunResult) -> Any`` that transforms the
      sub-agent result into the tool return value.  Defaults to extracting
      ``result.data``.
    inherit_memory_count: Number of parent memory entries to copy when
      ``memory_strategy`` is ``INHERIT``.
  """

  def __init__(
    self,
    agent: Agent,
    name: str | None = None,
    description: str | None = None,
    memory_strategy: MemoryStrategy | str = MemoryStrategy.ISOLATE,
    max_depth: int = 3,
    result_extractor: Callable[[RunResult], Any] | None = None,
    inherit_memory_count: int = 5,
  ):
    self.agent = agent
    self._name = name or getattr(agent, "name", None) or f"agent_{agent.model}"
    self._description = (
      description
      or f"Delegate tasks to a sub-agent powered by {agent.model}."
    )
    self.memory_strategy = (
      MemoryStrategy(memory_strategy)
      if isinstance(memory_strategy, str)
      else memory_strategy
    )
    self.max_depth = max_depth
    self.result_extractor = result_extractor or self._default_result_extractor
    self.inherit_memory_count = inherit_memory_count

  # -- Capability interface --------------------------------------------- #

  @property
  def name(self) -> str:
    return self._name

  @property
  def description(self) -> str:
    return self._description

  @property
  def parameters(self) -> dict[str, Any]:
    return {
      "type": "object",
      "properties": {
        "task": {
          "type": "string",
          "description": (
            "The task or question to delegate to the sub-agent. "
            "Be specific and include all necessary details."
          ),
        },
        "context": {
          "type": "string",
          "description": (
            "Optional additional context, background information, or "
            "constraints to pass to the sub-agent."
          ),
        },
      },
      "required": ["task"],
    }

  async def invoke(self, ctx: RunContext, arguments: dict[str, Any]) -> Any:
    """Execute the sub-agent and return its result.

    The execution flow:
    1. Check nesting depth — abort if limit exceeded.
    2. Check parent cancellation — abort if already cancelled.
    3. Resolve the ``Runner`` to use (from session or create a default).
    4. Build the effective prompt from ``task`` + optional ``context``.
    5. Run the sub-agent with the chosen memory strategy.
    6. Extract and return the result.
    """
    # 1. Depth guard
    current_depth = getattr(ctx.session, "_agent_depth", 0)
    if current_depth >= self.max_depth:
      return {
        "error": (
          f"Maximum agent nesting depth ({self.max_depth}) exceeded. "
          f"Current depth is {current_depth}. "
          "The sub-agent cannot be invoked at this depth."
        ),
      }

    # 2. Cancel propagation — respect parent cancellation
    if ctx.session.is_cancelled:
      return {
        "error": (
          "Sub-agent invocation cancelled: "
          "the parent session has been cancelled."
        ),
      }

    # 3. Resolve runner
    runner = self._resolve_runner(ctx)
    if runner is None:
      return {
        "error": (
          "No Runner available to execute the sub-agent. "
          "Ensure the parent session was created through a Runner, "
          "or pass a runner explicitly."
        ),
      }

    # 4. Build prompt
    task = arguments.get("task", "")
    extra_context = arguments.get("context", "")
    prompt = task
    if extra_context:
      prompt = f"{task}\n\nAdditional context:\n{extra_context}"

    # 5. Execute based on memory strategy
    if self.memory_strategy == MemoryStrategy.ISOLATE:
      result = await self._run_isolate(ctx, runner, prompt)
    elif self.memory_strategy == MemoryStrategy.INHERIT:
      result = await self._run_inherit(ctx, runner, prompt)
    elif self.memory_strategy == MemoryStrategy.SHARE:
      result = await self._run_share(ctx, runner, prompt)
    else:
      result = await self._run_isolate(ctx, runner, prompt)

    # 6. Extract result
    return self.result_extractor(result)

  def to_json_schema(self) -> dict[str, Any]:
    return {
      "type": "function",
      "function": {
        "name": self.name,
        "description": self.description,
        "parameters": self.parameters,
      },
    }

  # -- Internal helpers ------------------------------------------------- #

  @staticmethod
  def _default_result_extractor(result: RunResult) -> Any:
    """Default extraction: return ``result.data`` with error metadata."""
    if result.success:
      return result.data
    return {
      "error": result.error or "Sub-agent execution failed.",
      "error_type": result.error_type or "unknown",
      "success": False,
    }

  def _resolve_runner(self, ctx: RunContext) -> Any | None:
    """Try to obtain the ``Runner`` that created the parent session."""
    ref = getattr(ctx.session, "_runner_ref", None)
    if ref is not None:
      runner = ref()
      if runner is not None:
        return runner

    # Fallback: create a default runner.  This works but loses hooks,
    # checkpoint continuity, etc.  Log a warning so users know.
    _logger.warning(
      "agent_tool.fallback_runner",
      session_id=ctx.session_id,
      tool_name=self.name,
      message=(
        "No runner found on parent session; creating a default Runner. "
        "Hooks and checkpoint continuity will not be inherited."
      ),
    )
    from nonoka.core.runner import Runner

    return Runner(checkpoint="memory", memory="in_memory")

  async def _run_isolate(
    self,
    ctx: RunContext,
    runner: Any,
    prompt: str,
  ) -> RunResult:
    """Run sub-agent in a completely isolated session."""
    session = await runner._create_session(self.agent, ctx.deps)
    return await self._run_child(ctx, runner, session, prompt)

  async def _run_inherit(
    self,
    ctx: RunContext,
    runner: Any,
    prompt: str,
  ) -> RunResult:
    """Run sub-agent, copying the last N parent memory entries."""
    session = await runner._create_session(self.agent, ctx.deps)

    # Copy last N entries from parent memory
    if ctx.session.memory is not None and session.memory is not None:
      parent_entries = ctx.session.memory.entries
      to_copy = parent_entries[-self.inherit_memory_count :]
      for entry in to_copy:
        session.memory.entries.append(entry)

    return await self._run_child(ctx, runner, session, prompt)

  async def _run_share(
    self,
    ctx: RunContext,
    runner: Any,
    prompt: str,
  ) -> RunResult:
    """Run sub-agent sharing the parent's WorkingMemory object."""
    session = await runner._create_session(self.agent, ctx.deps)

    # Share the same WorkingMemory instance
    if ctx.session.memory is not None:
      object.__setattr__(session, "memory", ctx.session.memory)

    return await self._run_child(ctx, runner, session, prompt)

  def _lineage_key(self, prompt: str) -> str:
    """Stable key identifying one sub-agent invocation in the lineage map."""
    agent_id = getattr(self.agent, "name", None) or self.agent.model
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
    return f"{agent_id}:{prompt_hash}"

  async def _resume_child_session(
    self,
    ctx: RunContext,
    runner: Any,
    record: dict[str, Any],
  ) -> Any | None:
    """Rebuild a still-resumable child session from the checkpoint store.

    Mirrors ``Runner.resume``: load the state, recreate WorkingMemory when a
    memory backend is configured, then restore via ``Session.from_state``.
    Returns ``None`` when the child is missing or already terminal.
    """
    from nonoka.core.session import Session, SessionStatus

    child_session_id = record.get("child_session_id")
    if not child_session_id:
      return None
    state = await runner.checkpoint_store.load_session(child_session_id)
    if state is None or state.status in {
      SessionStatus.COMPLETED, SessionStatus.FAILED, SessionStatus.CANCELLED,
    }:
      return None

    memory = None
    if getattr(runner, "memory_backend", None) is not None:
      from nonoka.core.memory import WorkingMemory
      resumed_limits = getattr(getattr(state, "runtime_state", None), "limits", None)
      summary_llm = (
        runner._ensure_llm(self.agent)
        if getattr(resumed_limits, "summary_enabled", False)
        else None
      )
      memory = WorkingMemory(
        session_id=child_session_id,
        memory_backend=runner.memory_backend,
        summary_llm=summary_llm,
      )
    return Session.from_state(state, self.agent, deps=ctx.deps, memory=memory)

  async def _run_child(
    self,
    ctx: RunContext,
    runner: Any,
    session: Any,
    prompt: str,
  ) -> RunResult:
    """Execute a child session with the parent Runner lifecycle intact.

    ``AgentTool`` historically invoked ``ReActAgent`` directly.  That skipped
    the child's configured model selection and the Runner's session hooks, and
    it only observed parent cancellation before starting.  Keep the lightweight
    tool-based architecture while preserving those Runner invariants here.

    A lineage record is kept in the parent's ``extension_state`` so a parent
    crash does not orphan the child session: a ``running`` record whose child
    is still resumable continues that session instead of starting a new one,
    and a ``completed`` record returns the cached result text (a stringified
    ``result.data``) without re-executing the sub-agent.
    """
    from nonoka.core.extensions import LoopExtensionContext, LoopExtensionManager
    from nonoka.core.hooks import HookContext
    from nonoka.core.paradigm import ReActAgent

    lineage_key = self._lineage_key(prompt)
    lineage = ctx.session.extension_state.setdefault("agent_tool_lineage", {})
    record = lineage.get(lineage_key)
    resumed = False
    if isinstance(record, dict):
      if record.get("status") == "completed":
        # The child finished but the parent crashed before consuming the
        # result — return the cached text instead of re-running.
        del lineage[lineage_key]
        await runner.checkpoint_store.save_session(
          ctx.session.session_id, ctx.session.to_state()
        )
        return RunResult(success=True, data=record.get("result_text"), session=session)
      if record.get("status") == "running":
        resumed_session = await self._resume_child_session(ctx, runner, record)
        if resumed_session is not None:
          session = resumed_session
          resumed = True
    if not resumed:
      lineage[lineage_key] = {
        "child_session_id": session.session_id,
        "status": "running",
        "result_text": None,
      }
      # Persist the lineage record so a later parent resume can find the
      # (possibly orphaned) child session.
      await runner.checkpoint_store.save_session(
        ctx.session.session_id, ctx.session.to_state()
      )

    object.__setattr__(
      session,
      "_agent_depth",
      getattr(ctx.session, "_agent_depth", 0) + 1,
    )
    object.__setattr__(session, "_parent_session_id", ctx.session_id)

    previous_llm = getattr(runner, "llm", None)
    active_llm_token = None
    activate_agent = getattr(runner, "activate_agent", None)
    if callable(activate_agent):
      active_llm_token = activate_agent(self.agent)
    else:
      ensure_llm = getattr(runner, "_ensure_llm", None)
      if callable(ensure_llm):
        ensure_llm(self.agent)

    hook_ctx = HookContext(session=session, runner=runner)
    paradigm = ReActAgent()
    invocation_task = asyncio.current_task()
    parent_cancel_event = getattr(ctx.session, "_cancel_event", None)

    async def _propagate_parent_cancellation() -> None:
      await parent_cancel_event.wait()
      session.cancel()
      if invocation_task is not None:
        invocation_task.cancel()

    cancel_task = (
      asyncio.create_task(_propagate_parent_cancellation())
      if parent_cancel_event
      else None
    )
    result: RunResult
    try:
      await runner.hooks.emit_session_start(hook_ctx)
      if resumed:
        result = await paradigm.resume(session, runner)
      else:
        result = await paradigm.run(session, runner, prompt=prompt)

      await LoopExtensionManager(list(getattr(self.agent, "extensions", []))).after_run(
        LoopExtensionContext(
          session=session,
          runner=runner,
          prompt=prompt,
          turn=session.turn_count,
        ),
        result,
      )
      attach_trace = getattr(runner, "_attach_trace", None)
      if callable(attach_trace):
        result = attach_trace(result)
      if result.success:
        record = lineage.get(lineage_key)
        if isinstance(record, dict):
          record["status"] = "completed"
          record["result_text"] = str(result.data)
          await runner.checkpoint_store.save_session(
            ctx.session.session_id, ctx.session.to_state()
          )
      return result
    except asyncio.CancelledError:
      session.cancel()
      await runner.checkpoint_store.save_session(session.session_id, session.to_state())
      record = lineage.get(lineage_key)
      if isinstance(record, dict):
        record["status"] = "cancelled"
        await runner.checkpoint_store.save_session(
          ctx.session.session_id, ctx.session.to_state()
        )
      result = RunResult(
        success=False,
        session=session,
        error=(
          "Sub-agent invocation cancelled with its parent session."
          if ctx.session.is_cancelled
          else "Sub-agent invocation cancelled."
        ),
        error_type="cancelled",
      )
      return result
    except Exception as exc:
      result = RunResult(
        success=False,
        session=session,
        error=str(exc),
        error_type=type(exc).__name__,
      )
      return result
    finally:
      if cancel_task is not None:
        cancel_task.cancel()
        with suppress(asyncio.CancelledError):
          await cancel_task
      try:
        final_result = locals().get("result")
        if isinstance(final_result, RunResult):
          await runner.hooks.emit_session_end(hook_ctx, final_result)
      finally:
        reset_active_llm = getattr(runner, "reset_active_llm", None)
        if active_llm_token is not None and callable(reset_active_llm):
          reset_active_llm(active_llm_token)
        runner.llm = previous_llm
