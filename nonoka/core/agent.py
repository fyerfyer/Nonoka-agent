from dataclasses import dataclass, field
from typing import Generic, TypeVar, Any

from nonoka.core.types import Capability, RetryPolicy, RunResult

DepsT = TypeVar("DepsT")
ResultT = TypeVar("ResultT")


@dataclass(frozen=True)
class Agent(Generic[DepsT, ResultT]):
  """
  Agent is a state-less, immutable configuration object.

  It holds the model, tools, and execution policy.  Runtime state
  (plan progress, checkpoint, memory) lives in ``Session``.

  Agent does **not** execute directly — use ``Runner`` to choose an
  execution paradigm (``run_react``, ``run_plan``, ``run_reflective``).

  ``tools`` accepts either a list of ``Capability`` objects or a
  ``ToolRegistry`` (or a mix of both).  Registries are expanded at
  construction time so the Agent remains a pure data object.

  Usage::

    from nonoka import Agent, tool, Runner, ToolRegistry

    registry = ToolRegistry()

    @registry.register
    async def get_weather(city: str) -> dict: ...

    agent = Agent(model="gpt-4o", tools=registry)
    runner = Runner()
    result = await runner.run_react(agent, "What's the weather in Beijing?")
  """
  model: str
  tools: list[Capability] | "ToolRegistry" = field(default_factory=list)
  system_prompt: str = ""

  # Generic type hints for runtime type inference
  deps_type: type[DepsT] | None = None
  result_type: type[ResultT] | None = None

  # Default execution policy
  max_turns: int = 10
  max_steps: int = 50
  max_concurrency: int = 10  # Max concurrent tool calls within a single turn
  default_retry: RetryPolicy = field(default_factory=RetryPolicy)
  default_timeout: float | None = None

  # Metadata for routing, observability, and platform integration
  metadata: dict[str, Any] = field(default_factory=dict)
  tags: list[str] = field(default_factory=list)

  def __post_init__(self):
    """Expand any ``ToolRegistry`` values in *tools* to plain capabilities."""
    from nonoka.core.registry import ToolRegistry

    flat_tools: list[Capability] = []
    has_registry = False
    for item in self.tools if not isinstance(self.tools, ToolRegistry) else [self.tools]:
      if isinstance(item, ToolRegistry):
        has_registry = True
        flat_tools.extend(item.get_all())
      else:
        flat_tools.append(item)

    if has_registry or isinstance(self.tools, ToolRegistry):
      # Frozen dataclass — use object.__setattr__ to mutate once during init.
      object.__setattr__(self, "tools", flat_tools)

  async def run(
    self,
    prompt: str,
    deps: DepsT | None = None,
    **runner_kwargs: Any,
  ) -> RunResult[ResultT]:
    """Convenience shortcut: create a default Runner and execute in ReAct mode.

    Args:
      prompt: The user prompt / task description.
      deps: Optional dependency object injected into tools.
      **runner_kwargs: Passed to ``Runner`` constructor (e.g. ``checkpoint="redis"``).
    """
    from nonoka.core.runner import Runner
    runner = Runner(**runner_kwargs)
    return await runner.run_react(self, prompt, deps)
