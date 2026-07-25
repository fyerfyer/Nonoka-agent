"""Optional coding enhancements built on the constrained core extension API."""

from __future__ import annotations

import json
import re
from typing import Any

from nonoka.core.extensions import ExtensionDecision, LoopExtensionContext
from nonoka.core.types import RunResult


class VerifierRepairExtension:
  """Request a bounded repair only after a deterministic verifier fails.

  The evaluator is the same protocol used by ``ReflectiveAgent``. Unlike the
  older wrapper paradigm, this extension keeps a single ReAct session and
  makes every verifier decision visible in the execution trace.
  """

  name = "verifier_repair"

  def __init__(self, evaluator: Any, max_repairs: int | None = 2) -> None:
    self.evaluator = evaluator
    self.max_repairs = None if max_repairs is None else max(0, max_repairs)

  async def before_final_answer(self, context: LoopExtensionContext) -> ExtensionDecision:
    result = RunResult(success=True, data=context.content or "", session=context.session)
    evaluation = await self.evaluator.evaluate(result)
    context.session.trace.record_verification(
      source=self.name,
      passed=evaluation.passed,
      score=evaluation.score,
      feedback=evaluation.feedback,
      details=getattr(evaluation, "details", {}),
    )
    if evaluation.passed:
      return ExtensionDecision(details={"passed": True})

    attempts = getattr(context.session, "_extension_repair_attempts", {})
    attempt = int(attempts.get(self.name, 0))
    if self.max_repairs is not None and attempt >= self.max_repairs:
      return ExtensionDecision(
        failure=(
          f"Verifier rejected the final answer after {attempt} repair attempt(s): "
          f"{evaluation.feedback or 'no verifier feedback'}"
        ),
        details={"passed": False, "attempt": attempt, "exhausted": True},
      )
    attempts[self.name] = attempt + 1
    context.session._extension_repair_attempts = attempts
    feedback = evaluation.feedback or "Verification failed. Repair the workspace and verify it before completing."
    attempt_label = (
      str(attempt + 1)
      if self.max_repairs is None
      else f"{attempt + 1}/{self.max_repairs}"
    )
    return ExtensionDecision(
      feedback=f"[Verifier feedback — repair attempt {attempt_label}]\n{feedback}",
      continue_loop=True,
      details={"passed": False, "attempt": attempt + 1},
    )


class ResponseGroundingExtension:
  """Validate a proposed final reply against the state established by tools.

  ``validator`` receives ``(context, content)`` and may return ``True`` for
  acceptance, ``False`` for a generic correction, or a non-empty string with
  specific corrective feedback. It is intentionally domain-agnostic, so a τ³
  adapter can ground a customer reply without placing retail policy in core.
  """

  name = "response_grounding"

  def __init__(self, validator: Any, max_repairs: int = 1) -> None:
    self.validator = validator
    self.max_repairs = max(0, max_repairs)

  async def before_final_answer(self, context: LoopExtensionContext) -> ExtensionDecision:
    value = self.validator(context, context.content or "")
    if hasattr(value, "__await__"):
      value = await value
    if value is True:
      return ExtensionDecision(details={"grounded": True})
    feedback = value if isinstance(value, str) and value else (
      "Your final response is inconsistent with verified tool state. Correct it using the tool evidence."
    )
    attempts = getattr(context.session, "_extension_grounding_attempts", {})
    attempt = int(attempts.get(self.name, 0))
    if attempt >= self.max_repairs:
      return ExtensionDecision(
        failure=f"Final response failed grounding validation: {feedback}",
        details={"grounded": False, "attempt": attempt, "exhausted": True},
      )
    attempts[self.name] = attempt + 1
    context.session._extension_grounding_attempts = attempts
    return ExtensionDecision(
      feedback=f"[Grounding feedback — revision {attempt + 1}/{self.max_repairs}]\n{feedback}",
      continue_loop=True,
      details={"grounded": False, "attempt": attempt + 1},
    )


class WorkspaceProgressExtension:
  """Guide an effect-required terminal task through explore/change/verify/stop.

  Command text is used only for the early exploration reminder. Completion
  guidance relies on checkpointed, host-attested runtime evidence, so the
  extension never claims success from a tool name, task name, or shell pattern.
  """

  name = "workspace_progress"

  def __init__(
    self,
    max_exploration_turns: int = 3,
    reminder_interval: int = 2,
    max_post_verification_batches: int = 2,
  ) -> None:
    self.max_exploration_turns = max(1, max_exploration_turns)
    self.reminder_interval = max(1, reminder_interval)
    self.max_post_verification_batches = max(1, max_post_verification_batches)

  async def after_tool_batch(self, context: LoopExtensionContext) -> ExtensionDecision:
    extension_state = getattr(context.session, "extension_state", None)
    if isinstance(extension_state, dict):
      state = extension_state.setdefault(
        self.name, {
          "exploration_turns": 0,
          "mutating": False,
          "effect_count": 0,
          "successful_command_count": 0,
          "verification_ready": False,
          "post_verification_batches": 0,
          "completion_reminded": False,
        }
      )
    else:
      # Compatibility for lightweight test doubles and older Session objects.
      state = getattr(
        context.session,
        "_workspace_progress",
        {
          "exploration_turns": 0,
          "mutating": False,
          "effect_count": 0,
          "successful_command_count": 0,
          "verification_ready": False,
          "post_verification_batches": 0,
          "completion_reminded": False,
        },
      )
    runtime_state = getattr(context.session, "runtime_state", None)
    usage = getattr(runtime_state, "usage", None)
    effect_count = int(getattr(usage, "effect_count", 0))
    successful_count = int(getattr(usage, "successful_command_count", 0))
    previous_effect_count = int(state.get("effect_count", 0))
    previous_successful_count = int(state.get("successful_command_count", 0))
    effect_advanced = effect_count > previous_effect_count
    success_advanced = successful_count > previous_successful_count

    commands = [_command_from_call(call) for call in context.tool_calls]
    if effect_advanced or any(_looks_mutating(command) for command in commands):
      state["mutating"] = True
    elif not state["mutating"]:
      state["exploration_turns"] = int(state["exploration_turns"]) + 1

    if effect_advanced:
      state["verification_ready"] = False
      state["post_verification_batches"] = 0
      state["completion_reminded"] = False

    last_effect_at = getattr(usage, "last_effect_at_observation", None)
    last_success_at = getattr(usage, "last_successful_command_at_observation", None)
    last_complete_at = getattr(usage, "last_complete_observation_at", None)
    unresolved_partial = bool(
      getattr(usage, "last_partial_observation_at", None) is not None
      and (
        getattr(usage, "last_complete_observation_at", None) is None
        or getattr(usage, "last_complete_observation_at")
        <= getattr(usage, "last_partial_observation_at")
      )
    )
    post_effect_complete = bool(
      last_effect_at is not None
      and last_complete_at is not None
      and last_complete_at > last_effect_at
    )
    newly_ready = bool(
      effect_count > 0
      and not state.get("verification_ready")
      and last_effect_at is not None
      and (
        (
          success_advanced
          and last_success_at is not None
          and last_success_at > last_effect_at
        )
        or post_effect_complete
      )
      and not unresolved_partial
    )
    if newly_ready:
      state["verification_ready"] = True
      state["post_verification_batches"] = 0
    elif state.get("verification_ready") and not effect_advanced:
      state["post_verification_batches"] = int(state.get("post_verification_batches", 0)) + 1

    state["effect_count"] = effect_count
    state["successful_command_count"] = successful_count
    if not isinstance(extension_state, dict):
      context.session._workspace_progress = state

    if newly_ready and not state.get("completion_reminded"):
      state["completion_reminded"] = True
      return ExtensionDecision(
        feedback=(
          "[Completion evidence] A host-observed task effect is followed by fresh, complete "
          "post-change evidence, with no unresolved partial observation. If the requested acceptance "
          "criteria are satisfied, stop now and provide the final answer. Do not create "
          "optional artifacts or rerun equivalent checks unless the latest evidence shows "
          "a concrete failure or an uncovered requirement."
        ),
        details={
          "phase": "verification_ready",
          "effect_count": effect_count,
          "successful_command_count": successful_count,
          "post_effect_complete": post_effect_complete,
          "verification_ready": True,
        },
      )

    post_verification_batches = int(state.get("post_verification_batches", 0))
    if state.get("verification_ready") and post_verification_batches >= self.max_post_verification_batches:
      state["post_verification_batches"] = 0
      return ExtensionDecision(
        feedback=(
          "[Verification budget] The task already has post-change success evidence, and "
          "additional tool batches have not produced a new effect. Finish now unless you "
          "can name a specific unmet acceptance criterion; otherwise perform only the one "
          "smallest check needed for that criterion."
        ),
        details={
          "phase": "verification_ready",
          "effect_count": effect_count,
          "verification_ready": True,
          "verification_budget_reached": True,
        },
      )

    exploration_turns = int(state["exploration_turns"])
    due = exploration_turns >= self.max_exploration_turns and (
      (exploration_turns - self.max_exploration_turns) % self.reminder_interval == 0
    )
    if state["mutating"] or not due:
      return ExtensionDecision(details={
        "phase": "implementation" if state["mutating"] else "exploration",
        "exploration_turns": exploration_turns,
        "mutation_command_seen": state["mutating"],
        "effect_count": effect_count,
        "verification_ready": bool(state.get("verification_ready")),
      })
    return ExtensionDecision(
      feedback=(
        "[Execution phase] This task requires a workspace change, but no host-observed effect exists after "
        "focused investigation. Stop broad exploration and do not repeat an already slow or equivalent read-only "
        "check with a longer timeout or wrapper. Select the best-supported candidate, make the smallest necessary "
        "change, then run one focused verification of that change."
      ),
      details={
        "phase": "exploration",
        "exploration_turns": exploration_turns,
        "mutation_command_seen": False,
        "effect_count": effect_count,
        "reminded": True,
      },
    )


def _command_from_call(call: dict[str, Any]) -> str:
  function = call.get("function", call)
  arguments = function.get("arguments", {}) if isinstance(function, dict) else {}
  if isinstance(arguments, str):
    try:
      arguments = json.loads(arguments)
    except ValueError:
      return ""
  return str(arguments.get("command", "")) if isinstance(arguments, dict) else ""


def _looks_mutating(command: str) -> bool:
  lowered = command.lower()
  markers = ("sed -i", "perl -i", "tee ", "git apply", "patch ", "cp ", "mv ", "rm ", "touch ")
  if any(marker in lowered for marker in markers):
    return True
  # ``2>/dev/null`` and ``2>&1`` are read-only stderr redirects. Count only
  # shell output redirects that are not prefixed by an fd and not directed to
  # a null device or another descriptor.
  return bool(re.search(r"(?<![0-9])>{1,2}(?!\s*(?:/dev/null|&?[0-9]))", lowered))
