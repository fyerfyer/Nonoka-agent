from __future__ import annotations

import weakref
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

    return Runner()

  async def _run_isolate(
    self,
    ctx: RunContext,
    runner: Any,
    prompt: str,
  ) -> RunResult:
    """Run sub-agent in a completely isolated session."""
    session = await runner._create_session(self.agent, ctx.deps)
    object.__setattr__(session, "_agent_depth", getattr(ctx.session, "_agent_depth", 0) + 1)

    from nonoka.core.paradigm import ReActAgent

    paradigm = ReActAgent()
    result = await paradigm.run(session, runner, prompt=prompt)
    return result

  async def _run_inherit(
    self,
    ctx: RunContext,
    runner: Any,
    prompt: str,
  ) -> RunResult:
    """Run sub-agent, copying the last N parent memory entries."""
    session = await runner._create_session(self.agent, ctx.deps)
    object.__setattr__(session, "_agent_depth", getattr(ctx.session, "_agent_depth", 0) + 1)

    # Copy last N entries from parent memory
    if ctx.session.memory is not None and session.memory is not None:
      parent_entries = ctx.session.memory.entries
      to_copy = parent_entries[-self.inherit_memory_count :]
      for entry in to_copy:
        session.memory.entries.append(entry)

    from nonoka.core.paradigm import ReActAgent

    paradigm = ReActAgent()
    result = await paradigm.run(session, runner, prompt=prompt)
    return result

  async def _run_share(
    self,
    ctx: RunContext,
    runner: Any,
    prompt: str,
  ) -> RunResult:
    """Run sub-agent sharing the parent's WorkingMemory object."""
    session = await runner._create_session(self.agent, ctx.deps)
    object.__setattr__(session, "_agent_depth", getattr(ctx.session, "_agent_depth", 0) + 1)

    # Share the same WorkingMemory instance
    if ctx.session.memory is not None:
      object.__setattr__(session, "memory", ctx.session.memory)

    from nonoka.core.paradigm import ReActAgent

    paradigm = ReActAgent()
    result = await paradigm.run(session, runner, prompt=prompt)
    return result
