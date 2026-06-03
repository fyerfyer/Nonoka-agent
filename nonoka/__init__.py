"""Nonoka — Deterministic orchestration meets conversational execution."""

from nonoka.core.agent import Agent
from nonoka.core.tool import tool
from nonoka.core.context import RunContext
from nonoka.core.types import RunResult, RetryPolicy
from nonoka.core.plan import Plan, Step, PlanBuilder, ref
from nonoka.core.errors import (
  AgentError,
  TransientError,
  SchemaError,
  LogicError,
  SafetyError,
  ResourceError,
  MaxTurnsExceeded,
)
__all__ = [
  "Agent",
  "tool",
  "RunContext",
  "RunResult",
  "RetryPolicy",
  "Plan",
  "Step",
  "PlanBuilder",
  "ref",
  "AgentError",
  "TransientError",
  "SchemaError",
  "LogicError",
  "SafetyError",
  "ResourceError",
  "MaxTurnsExceeded",
]


def __getattr__(name: str):
  """Lazy import of Runner to keep ``import nonoka`` lightweight."""
  if name == "Runner":
    from nonoka.core.runner import Runner
    return Runner
  raise AttributeError(f"module 'nonoka' has no attribute '{name}'")