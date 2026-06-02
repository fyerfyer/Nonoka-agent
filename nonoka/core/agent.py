from dataclasses import dataclass, field
from typing import Generic, TypeVar

from nonoka.core.plan import Plan
from nonoka.core.types import Capability, RetryPolicy, RunResult

DepsT = TypeVar("DepsT")
ResultT = TypeVar("ResultT")


@dataclass(frozen=True)
class Agent(Generic[DepsT, ResultT]):
  """
  Agent is a state-less, immutable configuration object.

  It holds the model, tools, and execution policy.  Runtime state
  (plan progress, checkpoint, memory) lives in ``Session``.

  Usage:

    from nonoka import Agent, tool

    @tool
    async def get_weather(city: str) -> dict: ...

    agent = Agent(model="gpt-4o", tools=[get_weather])
    result = await agent.run("What's the weather in Beijing?")
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

  # ------------------------------------------------------------------ #
  # Quick-start shortcuts (Level-1 API)
  # ------------------------------------------------------------------ #

  async def run(
    self,
    prompt: str,
    deps: DepsT | None = None,
  ) -> "RunResult[ResultT]":
    """Create a default Runner and execute in auto-detected mode.

    This is the simplest entry-point for one-off executions::

        result = await agent.run("What's the weather in Beijing?")
    """
    from nonoka.core.runner import Runner
    runner = Runner(model=self.model)
    return await runner.run(self, prompt, deps)

  async def run_chat(
    self,
    prompt: str,
    deps: DepsT | None = None,
  ) -> "RunResult[ResultT]":
    """Force conversational (ReAct) mode."""
    from nonoka.core.runner import Runner
    runner = Runner(model=self.model)
    return await runner.run_chat(self, prompt, deps)

  async def run_plan(
    self,
    plan: "Plan",
    deps: DepsT | None = None,
  ) -> "RunResult[ResultT]":
    """Execute a user-defined Plan via DAGScheduler."""
    from nonoka.core.runner import Runner
    runner = Runner(model=self.model)
    return await runner.run_plan(self, plan, deps)
