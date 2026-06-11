from __future__ import annotations

"""Nonoka — Deterministic orchestration meets conversational execution."""

from nonoka.core.agent import Agent
from nonoka.core.agent_tool import AgentTool, MemoryStrategy
from nonoka.core.tool import tool
from nonoka.core.context import RunContext
from nonoka.core.types import RunResult, RetryPolicy
from nonoka.core.plan import Plan, Step, PlanBuilder, ref
from nonoka.core.hooks import Hooks, HookContext
from nonoka.core.errors import (
  AgentError,
  TransientError,
  SchemaError,
  LogicError,
  SafetyError,
  ResourceError,
  MaxTurnsExceeded,
)
from nonoka.core.paradigm import (
  ReActAgent,
  PlanExecutor,
  ReflectiveAgent,
  EvaluationResult,
  ToolEvaluator,
)
from nonoka.core.prompt import (
  prompt,
  PromptTemplate,
  PartialPromptTemplate,
  PromptFunction,
)
from nonoka.core.tool_response import (
  ToolResponse,
  make_tool_response,
  unwrap_tool_response,
)
from nonoka.core.system_prompts import SystemPromptTemplate
from nonoka.config import Config, ConfigLoadError
from nonoka.core.builder import AgentBuilder, RunnerBuilder

__all__ = [
  # Core
  "Agent",
  "AgentTool",
  "MemoryStrategy",
  "tool",
  "RunContext",
  "RunResult",
  "RetryPolicy",
  "Plan",
  "Step",
  "PlanBuilder",
  "ref",
  "Hooks",
  "HookContext",
  "AgentError",
  "TransientError",
  "SchemaError",
  "LogicError",
  "SafetyError",
  "ResourceError",
  "MaxTurnsExceeded",
  "ReActAgent",
  "PlanExecutor",
  "ReflectiveAgent",
  "EvaluationResult",
  "ToolEvaluator",
  "prompt",
  "PromptTemplate",
  "PartialPromptTemplate",
  "PromptFunction",
  "ToolResponse",
  "make_tool_response",
  "unwrap_tool_response",
  "SystemPromptTemplate",
  # Config system (new)
  "Config",
  "ConfigLoadError",
  "AgentBuilder",
  "RunnerBuilder",
  # Hot reload
  "PluginManager",
]


def __getattr__(name: str):
  """Lazy import of Runner to keep ``import nonoka`` lightweight."""
  if name == "Runner":
    from nonoka.core.runner import Runner
    return Runner
  if name == "ToolRegistry":
    from nonoka.core.registry import ToolRegistry
    return ToolRegistry
  if name == "PluginManager":
    from nonoka.core.hot_reload import PluginManager
    return PluginManager
  raise AttributeError(f"module 'nonoka' has no attribute '{name}'")
