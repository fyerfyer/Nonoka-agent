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

  Usage::

    from nonoka import Agent, tool, Runner

    @tool
    async def get_weather(city: str) -> dict: ...

    agent = Agent(model="gpt-4o", tools=[get_weather])
    runner = Runner()
    result = await runner.run_react(agent, "What's the weather in Beijing?")
  """
  model: str
  tools: list[Capability] = field(default_factory=list)
  system_prompt: str = ""

  # Generic type hints for runtime type inference
  deps_type: type[DepsT] | None = None
  result_type: type[ResultT] | None = None

  # Default execution policy
  max_turns: int = 10
  max_steps: int = 50
  default_retry: RetryPolicy = field(default_factory=RetryPolicy)
  default_timeout: float | None = None

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
    runner = Runner(model=self.model, **runner_kwargs)
    return await runner.run_react(self, prompt, deps)
