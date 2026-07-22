"""Optional coding enhancements built on the constrained core extension API."""

from __future__ import annotations

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
