from __future__ import annotations

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
  temperature: float | None = None
  max_tokens: int | None = None
  default_retry: RetryPolicy = field(default_factory=RetryPolicy)
  default_timeout: float | None = None

  # Optional bounded enhancements executed by ReAct.  They may provide
  # feedback and final-answer validation but cannot override core tool safety.
  extensions: list[Any] = field(default_factory=list)

  # Skills — pre-configured capability packages expanded at construction time
  skills: list["Skill"] = field(default_factory=list)

  # Metadata for routing, observability, and platform integration
  metadata: dict[str, Any] = field(default_factory=dict)
  tags: list[str] = field(default_factory=list)

  def __post_init__(self):
    """Expand any ``ToolRegistry`` values in *tools* and apply *skills*.

    1. ``ToolRegistry`` values are replaced by a ``ToolListProxy`` so that
       runtime mutations (hot reload) are visible.
    2. ``skills`` are merged into *tools*, *system_prompt*, and *metadata*.
       The resulting Agent has ``skills=[]`` to avoid double-expansion.
    """
    from nonoka.core.registry import ToolRegistry
    from nonoka.core.hot_reload import ToolListProxy

    flat_tools: list[Capability] = []
    registries: list[ToolRegistry] = []
    for item in self.tools if not isinstance(self.tools, ToolRegistry) else [self.tools]:
      if isinstance(item, ToolRegistry):
        registries.append(item)
      else:
        flat_tools.append(item)

    if registries:
      # Use a proxy so the registry can be mutated at runtime (hot reload)
      proxy = ToolListProxy(flat_tools, registries)
      object.__setattr__(self, "tools", proxy)
    # If no registries were provided leave *tools* exactly as-is.

    # ------------------------------------------------------------------ #
    # Expand skills
    # ------------------------------------------------------------------ #
    if self.skills:
      self._expand_skills()
    elif self.metadata.get("_skill_manager") or self.metadata.get("_external_mcp_registry"):
      self._expand_skills_lazy()

  def _expand_skills(self) -> None:
    """Merge skills into tools, system_prompt, and metadata."""
    from nonoka.core.hot_reload import ToolListProxy

    # Resolve current tools (may be ToolListProxy after registry expansion)
    current_tools: list[Capability]
    if isinstance(self.tools, ToolListProxy):
      current_tools = list(self.tools)
    else:
      current_tools = list(self.tools)

    # Build merged tool map: Agent explicit tools have highest priority.
    # Skills are applied in list order; later skills override earlier ones.
    tool_map: dict[str, Capability] = {}
    for skill in self.skills:
      for tool in skill.tools:
        tool_map[tool.name] = tool
    for tool in current_tools:
      tool_map[tool.name] = tool

    merged_tools = list(tool_map.values())

    # Merge system prompts: agent + each skill (system_prompt then activation_prompt)
    parts: list[str] = []
    if self.system_prompt:
      parts.append(self.system_prompt)
    for skill in self.skills:
      if skill.system_prompt:
        parts.append(skill.system_prompt)
      if skill.activation_prompt:
        parts.append(skill.activation_prompt)
    merged_system_prompt = "\n\n".join(parts)

    # Merge metadata: skill metadata takes precedence over Agent metadata
    merged_metadata = dict(self.metadata)
    for skill in self.skills:
      merged_metadata.update(skill.metadata)

    object.__setattr__(self, "tools", merged_tools)
    object.__setattr__(self, "system_prompt", merged_system_prompt)
    object.__setattr__(self, "metadata", merged_metadata)
    object.__setattr__(self, "skills", [])

  def _expand_skills_lazy(self) -> None:
    """Lazy skill expansion: register tools but keep guidance on-demand.

    The skill registry block (name + description) is injected into the system
    prompt so the model knows which skills are available. The full skill
    guidance is loaded via the ``load_skill`` tool when needed.

    Also merges any host-managed external MCP registry so that external MCP
    tools appear alongside skill tools with consistent namespacing.
    """
    from nonoka.core.hot_reload import ToolListProxy
    from nonoka.core.external_mcp import ExternalMCPRegistry
    from nonoka.skills.registry import SkillRegistry

    skill_registry = self.metadata.get("_skill_manager")
    mcp_registry = self.metadata.get("_external_mcp_registry")
    has_skill_registry = isinstance(skill_registry, SkillRegistry)
    has_mcp_registry = isinstance(mcp_registry, ExternalMCPRegistry)

    if not has_skill_registry and not has_mcp_registry:
      return

    current_tools: list[Capability]
    if isinstance(self.tools, ToolListProxy):
      current_tools = list(self.tools)
    else:
      current_tools = list(self.tools)

    tool_map: dict[str, Capability] = {}
    # Skill tools are merged first; explicit agent tools override them.
    if has_skill_registry:
      for tool in skill_registry.get_tools():
        tool_map[tool.name] = tool
    # External MCP tools are merged next; explicit agent tools override them.
    if has_mcp_registry:
      for tool in mcp_registry.get_tools():
        tool_map[tool.name] = tool
    for tool in current_tools:
      tool_map[tool.name] = tool

    parts: list[str] = []
    if self.system_prompt:
      parts.append(self.system_prompt)
    if has_skill_registry:
      skill_block = skill_registry.build_registry_block()
      if skill_block:
        parts.append(skill_block)
    if has_mcp_registry:
      mcp_block = mcp_registry.build_registry_block()
      if mcp_block:
        parts.append(mcp_block)
    merged_system_prompt = "\n\n".join(parts)

    object.__setattr__(self, "tools", list(tool_map.values()))
    object.__setattr__(self, "system_prompt", merged_system_prompt)
    object.__setattr__(self, "skills", [])

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
    from nonoka.config.resolver import _resolve_tool_entry

    data = dict(data)  # shallow copy
    tools_raw = data.pop("tools", [])
    resolved_tools = [_resolve_tool_entry(t) for t in tools_raw]

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
