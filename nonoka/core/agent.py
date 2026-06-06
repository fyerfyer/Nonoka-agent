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

  # -- Config loading ------------------------------------------------------

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> "Agent[DepsT, ResultT]":
    """Construct an ``Agent`` from a plain dictionary.

    This is the programmatic counterpart to YAML/JSON config files.
    Tools specified as ``"module:function"`` strings are resolved
    automatically.

    Args:
      data: Dictionary with keys matching ``Agent`` field names.

    Example::

      agent = Agent.from_dict({
        "model": "gpt-4o",
        "system_prompt": "You are helpful.",
        "tools": ["my_tools:get_weather"],
        "max_turns": 20,
        "metadata": {"category": "weather"},
      })
    """
    from nonoka.core.config_loader import AgentConfig
    from nonoka.core.tool import tool as make_tool

    data = dict(data)  # shallow copy
    tools_raw = data.pop("tools", [])
    resolved_tools = []

    for t in tools_raw:
      if isinstance(t, str):
        # Import path — resolve it
        from nonoka.core.config_loader import resolve_tool_import
        obj = resolve_tool_import(t)
        if isinstance(obj, Capability):
          resolved_tools.append(obj)
        elif callable(obj):
          resolved_tools.append(make_tool(obj))
        else:
          raise TypeError(
            f"Tool import '{t}' resolved to {type(obj).__name__}, "
            "expected a callable or Capability"
          )
      elif isinstance(t, Capability):
        resolved_tools.append(t)
      elif callable(t):
        resolved_tools.append(make_tool(t))
      else:
        raise TypeError(f"Invalid tool entry: {t!r}")

    # Handle retry as a dict
    retry_raw = data.pop("default_retry", None)
    if isinstance(retry_raw, dict):
      data["default_retry"] = RetryPolicy(**retry_raw)

    return cls(tools=resolved_tools, **data)

  @classmethod
  def from_yaml(cls, path: str) -> "Agent[DepsT, ResultT]":
    """Construct an ``Agent`` from a YAML file.

    The YAML should contain a single Agent definition (not the full
    ``agents:`` dictionary).

    Example YAML::

      model: gpt-4o
      system_prompt: "You are helpful."
      tools:
        - import: my_tools:get_weather
      max_turns: 20
    """
    try:
      import yaml
    except ImportError as exc:
      raise ImportError("PyYAML is required. Install: pip install pyyaml") from exc
    with open(path, encoding="utf-8") as f:
      data = yaml.safe_load(f)
    if not isinstance(data, dict):
      raise ValueError(f"YAML file must contain a dict, got {type(data).__name__}")
    return cls.from_dict(data)

  @classmethod
  def from_json(cls, path: str) -> "Agent[DepsT, ResultT]":
    """Construct an ``Agent`` from a JSON file."""
    import json
    with open(path, encoding="utf-8") as f:
      data = json.load(f)
    if not isinstance(data, dict):
      raise ValueError(f"JSON file must contain a dict, got {type(data).__name__}")
    return cls.from_dict(data)

  # -- Convenience run -------------------------------------------------------

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
