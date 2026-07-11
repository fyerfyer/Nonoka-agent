"""Fluent Builder API for constructing Agents and Runners."""

from __future__ import annotations

from typing import Any, TypeVar

from nonoka.core.types import Capability, RetryPolicy
from nonoka.core.runner import _Unset, _UNSET


DepsT = TypeVar("DepsT")
ResultT = TypeVar("ResultT")


class AgentBuilder:
  """Fluent builder for constructing ``Agent`` instances.

  Usage::

    from nonoka import AgentBuilder, ToolRegistry, tool

    @tool
    async def get_weather(city: str) -> str:
      return f"Sunny in {city}!"

    registry = ToolRegistry()

    @registry.register
    async def search_city(name: str) -> str:
      return f"Found {name}"

    agent = (
      AgentBuilder()
      .model("gpt-4o")
      .system_prompt("You are a weather assistant.")
      .tool(get_weather)
      .tool_registry(registry)
      .tool_by_import("my_tools.search:search_city")
      .skill(my_skill)
      .max_turns(20)
      .retry(max_retries=5, backoff=1.5)
      .timeout(45.0)
      .metadata(category="weather", region="apac")
      .tag("production")
      .build()
    )
  """

  def __init__(self):
    self._model: str | None = None
    self._tools: list[Capability] = []
    self._tool_registries: list[Any] = []
    self._skills: list[Any] = []
    self._skill_manager: Any | None = None
    self._system_prompt: str = ""
    self._max_turns: int | None = None
    self._max_steps: int | None = None
    self._max_concurrency: int | None = None
    self._default_retry: RetryPolicy | None = None
    self._default_timeout: float | None = None
    self._metadata: dict[str, Any] | None = None
    self._tags: list[str] | None = None
    self._deps_type: type | None = None
    self._result_type: type | None = None

  # -- Model & Prompt --------------------------------------------------------

  def model(self, value: str) -> AgentBuilder:
    """Set the LLM model identifier (e.g. ``gpt-4o``, ``deepseek-chat``)."""
    self._model = value
    return self

  def system_prompt(self, value: str) -> AgentBuilder:
    """Set the system prompt sent to the LLM."""
    self._system_prompt = value
    return self

  # -- Tools -----------------------------------------------------------------

  def tool(self, capability: Capability | Any) -> AgentBuilder:
    """Add a tool (``Capability`` or callable). Callables are auto-wrapped."""
    from nonoka.core.tool import tool as make_tool
    from nonoka.core.types import Capability as CapProtocol

    if isinstance(capability, CapProtocol):
      self._tools.append(capability)
    elif callable(capability):
      self._tools.append(make_tool(capability))
    else:
      raise TypeError(
        f"tool() expects a Capability or callable, got {type(capability).__name__}"
      )
    return self

  def tools(self, *capabilities: Capability | Any) -> AgentBuilder:
    """Add multiple tools at once.

    Accepts ``Capability`` instances, raw callables, or ``ToolRegistry``
    objects. Registries are kept separate so runtime mutations are visible
    to the built Agent.
    """
    from nonoka.core.registry import ToolRegistry

    for cap in capabilities:
      if isinstance(cap, ToolRegistry):
        self._tool_registries.append(cap)
      else:
        self.tool(cap)
    return self

  def tool_registry(self, registry: Any) -> AgentBuilder:
    """Add a ``ToolRegistry`` whose contents are expanded at build time.

    Registries are wrapped in a ``ToolListProxy`` so mutations made after
    the Agent is built are still visible on subsequent tool lookups.
    """
    self._tool_registries.append(registry)
    return self

  def tool_by_import(self, import_path: str) -> AgentBuilder:
    """Add a tool by import path: ``module.submodule:function_name``."""
    from nonoka.config.resolver import _resolve_tool_entry

    self._tools.append(_resolve_tool_entry(import_path))
    return self

  # -- Skills ----------------------------------------------------------------

  def skill(self, skill: Any) -> AgentBuilder:
    """Add a single ``Skill`` to be applied when the Agent is built."""
    self._skills.append(skill)
    return self

  def skills(self, *skills: Any) -> AgentBuilder:
    """Add multiple ``Skill`` objects at once."""
    self._skills.extend(skills)
    return self

  def skill_manager(self, manager: Any) -> AgentBuilder:
    """Set a SkillRegistry for lazy skill loading.

    When a skill manager is provided, skills are exposed via a lightweight
    registry block in the system prompt. Their tools are registered eagerly,
    but their full guidance is only loaded when the model calls the
    ``load_skill`` tool. This avoids bloating the system prompt with every
    skill's activation prompt.
    """
    self._skill_manager = manager
    return self

  # -- Execution policy ------------------------------------------------------

  def max_turns(self, value: int) -> AgentBuilder:
    """Set the maximum conversation turns."""
    self._max_turns = value
    return self

  def max_steps(self, value: int) -> AgentBuilder:
    """Set the maximum execution steps."""
    self._max_steps = value
    return self

  def max_concurrency(self, value: int) -> AgentBuilder:
    """Set the maximum concurrent tool calls per turn."""
    self._max_concurrency = value
    return self

  def retry(self, *, max_retries: int = 3, backoff: float = 2.0) -> AgentBuilder:
    """Set the default retry policy for LLM calls."""
    self._default_retry = RetryPolicy(max_retries=max_retries, backoff=backoff)
    return self

  def timeout(self, value: float | None) -> AgentBuilder:
    """Set the timeout (seconds) for LLM calls."""
    self._default_timeout = value
    return self

  # -- Metadata --------------------------------------------------------------

  def metadata(self, **kwargs: Any) -> AgentBuilder:
    """Set metadata key-value pairs (merged with any existing)."""
    if self._metadata is None:
      self._metadata = {}
    self._metadata.update(kwargs)
    return self

  def tag(self, *tags: str) -> AgentBuilder:
    """Add tags for categorization."""
    if self._tags is None:
      self._tags = []
    self._tags.extend(tags)
    return self

  def deps_type(self, t: type[DepsT]) -> AgentBuilder:
    """Set the dependency type hint."""
    self._deps_type = t
    return self

  def result_type(self, t: type[ResultT]) -> AgentBuilder:
    """Set the result type hint."""
    self._result_type = t
    return self

  # -- Build -----------------------------------------------------------------

  def build(self) -> Any:
    """Construct and return an ``Agent`` instance."""
    from nonoka.core.agent import Agent

    if self._model is None:
      raise ValueError("AgentBuilder: model is required. Call .model(...) before .build()")

    tools: list[Capability | Any] = [*self._tools, *self._tool_registries]

    kwargs: dict[str, Any] = {
      "model": self._model,
      "tools": tools,
      "system_prompt": self._system_prompt,
    }
    if self._max_turns is not None:
      kwargs["max_turns"] = self._max_turns
    if self._max_steps is not None:
      kwargs["max_steps"] = self._max_steps
    if self._max_concurrency is not None:
      kwargs["max_concurrency"] = self._max_concurrency
    if self._default_retry is not None:
      kwargs["default_retry"] = self._default_retry
    if self._default_timeout is not None:
      kwargs["default_timeout"] = self._default_timeout
    if self._metadata is not None:
      kwargs["metadata"] = self._metadata
    if self._tags is not None:
      kwargs["tags"] = self._tags
    if self._deps_type is not None:
      kwargs["deps_type"] = self._deps_type
    if self._result_type is not None:
      kwargs["result_type"] = self._result_type
    if self._skill_manager is not None:
      metadata = kwargs.setdefault("metadata", {})
      metadata["_skill_manager"] = self._skill_manager
    elif self._skills:
      kwargs["skills"] = self._skills

    return Agent(**kwargs)


class RunnerBuilder:
  """Fluent builder for constructing ``Runner`` instances.

  Usage::

    runner = (
      RunnerBuilder()
      .checkpoint("redis")
      .memory("in_memory")
      .build()
    )
  """

  def __init__(self):
    self._checkpoint: str | None | _Unset = _UNSET
    self._memory: str | None | _Unset = _UNSET
    self._circuit_breaker: Any | None = None
    self._hooks: Any | None = None
    self._gateway: Any | None = None

  def checkpoint(self, value: str | None) -> RunnerBuilder:
    """Set checkpoint backend (``memory``, ``disabled``, or custom)."""
    self._checkpoint = value
    return self

  def memory(self, value: str | None) -> RunnerBuilder:
    """Set memory backend (``in_memory``, ``disabled``, or custom)."""
    self._memory = value
    return self

  def circuit_breaker(self, value: Any | None) -> RunnerBuilder:
    """Set a shared circuit breaker."""
    self._circuit_breaker = value
    return self

  def hooks(self, value: Any | None) -> RunnerBuilder:
    """Set lifecycle hooks / middleware."""
    self._hooks = value
    return self

  def gateway(self, value: Any | None) -> RunnerBuilder:
    """Set gateway for reverse-channel push."""
    self._gateway = value
    return self

  def build(self) -> Any:
    """Construct and return a ``Runner`` instance."""
    from nonoka.core.runner import Runner

    kwargs: dict[str, Any] = {}
    if not isinstance(self._checkpoint, _Unset):
      kwargs["checkpoint"] = self._checkpoint
    if not isinstance(self._memory, _Unset):
      kwargs["memory"] = self._memory
    if self._circuit_breaker is not None:
      kwargs["circuit_breaker"] = self._circuit_breaker
    if self._hooks is not None:
      kwargs["hooks"] = self._hooks
    if self._gateway is not None:
      kwargs["gateway"] = self._gateway

    return Runner(**kwargs)
