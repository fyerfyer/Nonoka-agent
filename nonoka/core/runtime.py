"""Serializable session-wide runtime limits and termination diagnostics."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


class TerminalReason(str, Enum):
  TURN_BUDGET_EXHAUSTED = "turn_budget_exhausted"
  TOOL_BUDGET_EXHAUSTED = "tool_budget_exhausted"
  CONTEXT_BUDGET_EXHAUSTED = "context_budget_exhausted"
  DEADLINE_EXCEEDED = "deadline_exceeded"
  MODEL_TIMEOUT = "model_timeout"
  CANCELLED = "cancelled"
  COMPLETION_CONTRACT_UNMET = "completion_contract_unmet"
  EXECUTION_POLICY_VIOLATION = "execution_policy_violation"


class RuntimeLimits(BaseModel):
  """Hard limits that apply to an entire persisted session."""

  max_model_turns: int | None = Field(default=None, ge=0)
  max_tool_calls: int | None = Field(default=None, ge=0)
  wall_timeout_seconds: float | None = Field(default=None, gt=0)
  model_timeout_seconds: float | None = Field(default=None, gt=0)
  max_context_bytes: int | None = Field(default=None, ge=1)
  max_context_tokens: int | None = Field(default=None, ge=1)
  max_tool_messages: int | None = Field(default=None, ge=1)
  max_external_result_bytes: int | None = Field(default=None, ge=1)


class Termination(BaseModel):
  reason: TerminalReason
  message: str
  dimension: str | None = None
  limit: int | float | None = None
  used: int | float | None = None
  diagnostics: dict[str, Any] = Field(default_factory=dict)


class RuntimeUsage(BaseModel):
  model_turns: int = 0
  tool_calls: int = 0
  external_tool_calls: int = 0
  input_tokens: int = 0
  output_tokens: int = 0
  context_bytes: int = 0
  context_tokens: int = 0
  tool_messages: int = 0
  mutation_count: int = 0
  # A task effect is broader than a cwd mutation. Terminal tasks may install a
  # package, start a service, or update a repository outside the task root.
  # Hosts must attest those effects explicitly; core never infers them from a
  # task name or command string.
  effect_count: int = 0
  last_effect_at_observation: int | None = None
  first_mutation_at: datetime | None = None
  last_tool: str | None = None
  correction_count: int = 0
  changed_paths: list[str] = Field(default_factory=list)
  verified_commands: list[str] = Field(default_factory=list)
  successful_command_count: int = 0
  last_successful_command: str | None = None
  last_successful_command_at_observation: int | None = None
  policy_violation_count: int = 0
  policy_violations: list[str] = Field(default_factory=list)
  observation_count: int = 0
  partial_observation_count: int = 0
  unknown_observation_count: int = 0
  last_partial_observation_at: int | None = None
  last_complete_observation_at: int | None = None
  observation_feedback_after: int = 0
  complete_observations_after_partial: int = 0
  latest_partial_tool: str | None = None
  latest_partial_artifact_ref: str | None = None
  completion_warning_count: int = 0
  latest_completion_warning: str | None = None


class CompletionRule(BaseModel):
  """Serializable completion predicate evaluated against runtime evidence."""

  kind: str

  def unmet(self, usage: RuntimeUsage) -> str | None:
    raise NotImplementedError


class WorkspaceMutationRule(CompletionRule):
  kind: Literal["workspace_mutation"] = "workspace_mutation"
  minimum_count: int = Field(default=1, ge=1)

  def unmet(self, usage: RuntimeUsage) -> str | None:
    if usage.mutation_count >= self.minimum_count:
      return None
    return f"make at least {self.minimum_count} observed workspace mutation(s)"


class ObservedEffectRule(CompletionRule):
  """Require a host-attested durable task effect.

  This deliberately differs from ``WorkspaceMutationRule``: system-oriented
  terminal tasks can make their required change outside the current working
  directory. The Adapter owns that trust boundary and must return typed effect
  evidence; the Agent core does not inspect shell syntax.
  """

  kind: Literal["observed_effect"] = "observed_effect"
  minimum_count: int = Field(default=1, ge=1)

  def unmet(self, usage: RuntimeUsage) -> str | None:
    if usage.effect_count >= self.minimum_count:
      return None
    return f"produce at least {self.minimum_count} host-observed task effect(s)"


class PathsChangedRule(CompletionRule):
  kind: Literal["paths_changed"] = "paths_changed"
  paths: tuple[str, ...]

  def unmet(self, usage: RuntimeUsage) -> str | None:
    missing = [path for path in self.paths if path not in usage.changed_paths]
    return f"modify the required paths: {', '.join(missing)}" if missing else None


class CommandSucceededRule(CompletionRule):
  kind: Literal["command_succeeded"] = "command_succeeded"
  command: str

  def unmet(self, usage: RuntimeUsage) -> str | None:
    if self.command in usage.verified_commands:
      return None
    return f"run the verification command successfully: {self.command}"


class CompleteObservationRule(CompletionRule):
  """Require an exhaustive observation after the latest unresolved partial one."""

  kind: Literal["complete_observation"] = "complete_observation"

  def unmet(self, usage: RuntimeUsage) -> str | None:
    if usage.last_partial_observation_at is None:
      return None
    # A generic COMPLETE receipt before correction may be an unrelated tool
    # result. Require the model to receive the completion feedback first, then
    # gather fresh bounded evidence in that new epoch.
    if usage.observation_feedback_after == 0:
      return (
        "obtain a complete follow-up observation after the latest partial result; "
        "narrow the query or inspect a bounded artifact or region"
      )
    required_after = max(
      usage.observation_feedback_after,
      usage.last_partial_observation_at,
    )
    if (
      usage.last_complete_observation_at is not None
      and usage.last_complete_observation_at > required_after
    ):
      return None
    return (
      "obtain a complete follow-up observation after the latest partial result; "
      "narrow the query or inspect a bounded artifact or region"
    )


CompletionRuleType = Annotated[
  WorkspaceMutationRule | ObservedEffectRule | PathsChangedRule | CommandSucceededRule | CompleteObservationRule,
  Field(discriminator="kind"),
]


class CompletionContract(BaseModel):
  """Host-supplied, declarative evidence required before completion.

  ``rules`` is the extensible contract surface. The legacy convenience fields
  remain supported and are compiled into rules so existing callers keep their
  behavior without duplicating evaluation logic in ``Session``.
  """

  rules: tuple[CompletionRuleType, ...] = ()
  require_workspace_mutation: bool = False
  require_observed_effect: bool = False
  required_paths: tuple[str, ...] = ()
  verification_command: str | None = None
  require_complete_observations: bool = False
  max_corrections: int = Field(default=1, ge=0)
  enforcement: Literal["strict", "advisory"] = "strict"

  def effective_rules(self) -> tuple[CompletionRuleType, ...]:
    compiled: list[CompletionRuleType] = list(self.rules)
    if self.require_workspace_mutation:
      compiled.append(WorkspaceMutationRule())
    if self.require_observed_effect:
      compiled.append(ObservedEffectRule())
    if self.required_paths:
      compiled.append(PathsChangedRule(paths=self.required_paths))
    if self.verification_command:
      compiled.append(CommandSucceededRule(command=self.verification_command))
    if self.require_complete_observations:
      compiled.append(CompleteObservationRule())
    return tuple(compiled)

  def unmet_requirements(self, usage: RuntimeUsage) -> list[str]:
    return [message for rule in self.effective_rules() if (message := rule.unmet(usage))]

  def observation_review_unmet(self, usage: RuntimeUsage) -> bool:
    return any(
      isinstance(rule, CompleteObservationRule) and rule.unmet(usage) is not None
      for rule in self.effective_rules()
    )


class SessionRuntimeState(BaseModel):
  """Checkpoint-owned runtime state. Resumes must restore, never recreate, it."""

  limits: RuntimeLimits
  usage: RuntimeUsage = Field(default_factory=RuntimeUsage)
  started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
  deadline_at: datetime | None = None
  cancelled_at: datetime | None = None
  termination: Termination | None = None

  @classmethod
  def create(cls, limits: RuntimeLimits) -> "SessionRuntimeState":
    started = datetime.now(timezone.utc)
    deadline = (
      started + timedelta(seconds=limits.wall_timeout_seconds)
      if limits.wall_timeout_seconds is not None
      else None
    )
    return cls(limits=limits, started_at=started, deadline_at=deadline)

  @property
  def is_cancelled(self) -> bool:
    return self.cancelled_at is not None

  def remaining_seconds(self) -> float | None:
    if self.deadline_at is None:
      return None
    return max(0.0, (self.deadline_at - datetime.now(timezone.utc)).total_seconds())
