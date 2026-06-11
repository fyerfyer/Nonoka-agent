from __future__ import annotations

from typing import Any

from nonoka.core.hooks import Hooks, HookContext
from nonoka.core.errors import HumanRejectedError, ApprovalTimeoutError
from nonoka.ext.hitl.core import (
  HumanApprover,
  HumanCheckpoint,
  HumanDecision,
  ToolRule,
)


class HumanInTheLoopHooks(Hooks):
  """Hooks subclass that injects human approval at tool-call time.

  When a tool call matches one of the configured rules, the execution is
  paused and a ``HumanCheckpoint`` is created.  The ``HumanApprover``
  presents it to a human and returns their decision.

  Usage::

    from nonoka.ext.hitl import HumanInTheLoopHooks, CLIApprover, ToolRule

    hitl = HumanInTheLoopHooks(
      approver=CLIApprover(timeout=30.0),
      rules=[
        ToolRule(tool="edit_file", action="modify"),
        ToolRule(tool="delete_file", action="approve"),
        ToolRule(tool="run_command", pattern="rm|drop|delete", action="approve"),
      ],
    )

    runner = Runner(hooks=hitl)
    result = await runner.run_react(agent, "Refactor this project")

  Args:
    approver: Backend that presents checkpoints to humans.
    rules: List of ``ToolRule`` objects that decide which calls require
      approval.  If empty, no calls are intercepted.
    default_action: What to do when no rule matches.  ``"allow"`` (default)
      lets the call proceed without approval; ``"approve"`` forces approval
      for every call.
  """

  def __init__(
    self,
    approver: HumanApprover,
    rules: list[ToolRule] | None = None,
    default_action: str = "allow",
    **hooks_kwargs: Any,
  ):
    super().__init__(**hooks_kwargs)
    self.approver = approver
    self.rules = list(rules) if rules is not None else []
    self.default_action = default_action

  # -- Subclass hook override ------------------------------------------- #

  async def on_tool_start_intercept(
    self,
    ctx: HookContext,
    tool_name: str,
    arguments: dict[str, Any],
  ) -> dict[str, Any]:
    """Intercept tool calls and request human approval when rules match."""
    matched_rule = self._match_rule(tool_name, arguments)

    if matched_rule is None:
      if self.default_action == "approve":
        # Force approval even without a matching rule
        matched_rule = ToolRule(tool="*", action="approve")
      else:
        # No match and default is allow — pass through unchanged
        return arguments

    # Build checkpoint
    checkpoint = HumanCheckpoint(
      trigger=f"tool_call:{tool_name}",
      description=matched_rule.description or f"Tool '{tool_name}' requires human approval.",
      context={
        "session_id": ctx.session.session_id,
        "turn_count": ctx.session.turn_count,
        "step_count": ctx.session.step_count,
        "rule_action": matched_rule.action,
      },
      original_args=arguments,
    )

    # Request approval
    resolved = await self.approver.request_approval(checkpoint)

    # Apply decision
    if resolved.decision == HumanDecision.REJECT:
      # Approver should have already raised HumanRejectedError, but guard anyway
      raise HumanRejectedError(
        f"Human rejected tool call '{tool_name}': {resolved.feedback}"
      )

    if resolved.decision == HumanDecision.MODIFY:
      return resolved.effective_args

    # APPROVE — return original (or modified) args
    return resolved.effective_args

  # -- Internal helpers ------------------------------------------------- #

  def _match_rule(self, tool_name: str, arguments: dict[str, Any]) -> ToolRule | None:
    """Return the first matching rule, or ``None`` if no rule matches."""
    for rule in self.rules:
      if rule.matches(tool_name, arguments):
        return rule
    return None
