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

  def __init__(self, evaluator: Any, max_repairs: int = 2) -> None:
    self.evaluator = evaluator
    self.max_repairs = max(0, max_repairs)

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
    if attempt >= self.max_repairs:
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
    return ExtensionDecision(
      feedback=f"[Verifier feedback — repair attempt {attempt + 1}/{self.max_repairs}]\n{feedback}",
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
  """Nudge an explicitly mutation-required terminal task out of exploration.

  The extension never claims to prove a filesystem change: it observes command
  intent only. Callers must opt in when the task contract requires a workspace
  mutation, preserving the core loop's task-agnostic behaviour.
  """

  name = "workspace_progress"

  def __init__(self, max_exploration_turns: int = 3, reminder_interval: int = 2) -> None:
    self.max_exploration_turns = max(1, max_exploration_turns)
    self.reminder_interval = max(1, reminder_interval)

  async def after_tool_batch(self, context: LoopExtensionContext) -> ExtensionDecision:
    state = getattr(context.session, "_workspace_progress", {"exploration_turns": 0, "mutating": False})
    commands = [_command_from_call(call) for call in context.tool_calls]
    if any(_looks_mutating(command) for command in commands):
      state["mutating"] = True
    elif not state["mutating"]:
      state["exploration_turns"] = int(state["exploration_turns"]) + 1
    context.session._workspace_progress = state
    exploration_turns = int(state["exploration_turns"])
    due = exploration_turns >= self.max_exploration_turns and (
      (exploration_turns - self.max_exploration_turns) % self.reminder_interval == 0
    )
    if state["mutating"] or not due:
      return ExtensionDecision(details={"exploration_turns": exploration_turns, "mutation_command_seen": state["mutating"]})
    return ExtensionDecision(
      feedback=(
        "[Workspace progress] This task requires a workspace change, but no mutation command has been observed. "
        "Stop broad exploration: identify the requested target, make the smallest necessary edit, then verify the result."
      ),
      details={"exploration_turns": exploration_turns, "mutation_command_seen": False, "reminded": True},
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
