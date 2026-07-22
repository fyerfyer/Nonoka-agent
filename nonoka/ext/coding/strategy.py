"""Explicit routing for code tasks; no implicit model classifier is required."""

from __future__ import annotations

from enum import Enum
from typing import Any

from .extensions import VerifierRepairExtension


class CodeStrategy(str, Enum):
  DIRECT = "direct"
  TOOL_ASSISTED = "tool_assisted"
  VERIFIED_REPAIR = "verified_repair"


class CodeStrategyRouter:
  """Choose a code strategy from caller-known task capabilities.

  A caller must explicitly state whether a deterministic verifier exists.
  This avoids pretending that an LLM prompt classifier can safely infer when
  an expensive tool loop is beneficial.
  """

  def choose(self, *, deterministic_verifier: bool, requires_workspace: bool) -> CodeStrategy:
    if deterministic_verifier and requires_workspace:
      return CodeStrategy.VERIFIED_REPAIR
    if requires_workspace:
      return CodeStrategy.TOOL_ASSISTED
    return CodeStrategy.DIRECT

  def extensions_for(self, strategy: CodeStrategy, evaluator: Any | None) -> list[Any]:
    if strategy is CodeStrategy.VERIFIED_REPAIR:
      if evaluator is None:
        raise ValueError("verified_repair requires a deterministic evaluator")
      return [VerifierRepairExtension(evaluator)]
    return []
