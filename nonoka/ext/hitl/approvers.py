from __future__ import annotations

import asyncio
from typing import Any

from nonoka.core.errors import HumanRejectedError, ApprovalTimeoutError
from nonoka.ext.hitl.core import HumanApprover, HumanCheckpoint, HumanDecision


class CLIApprover(HumanApprover):
  """Terminal-based human approval backend.

  Presents the checkpoint in the terminal and reads the user's response
  from stdin.  Suitable for development and local testing.

  Args:
    timeout: Maximum seconds to wait for human input.  ``None`` = block forever.
    default_on_timeout: Decision to apply when timeout is reached.
      ``"reject"`` (default) or ``"approve"``.
  """

  def __init__(
    self,
    timeout: float | None = 60.0,
    default_on_timeout: str = "reject",
  ):
    self.timeout = timeout
    self.default_on_timeout = default_on_timeout

  async def request_approval(self, checkpoint: HumanCheckpoint) -> HumanCheckpoint:
    """Present checkpoint in the terminal and wait for human input."""
    import sys

    print(f"\n{'=' * 60}")
    print(f"  HUMAN-IN-THE-LOOP CHECKPOINT  [{checkpoint.checkpoint_id}]")
    print(f"{'=' * 60}")
    print(f"  Trigger:    {checkpoint.trigger}")
    print(f"  Description: {checkpoint.description}")
    print(f"  Context:    {checkpoint.context}")
    print(f"  Arguments:  {checkpoint.original_args}")
    print(f"{'=' * 60}")
    print("  [a]pprove  |  [m]odify  |  [r]eject")
    print(f"{'=' * 60}")

    try:
      response = await asyncio.wait_for(
        self._read_input(),
        timeout=self.timeout,
      )
    except asyncio.TimeoutError:
      print(f"\n[Timeout] No response within {self.timeout}s.")
      if self.default_on_timeout == "reject":
        raise ApprovalTimeoutError(
          f"Approval timed out after {self.timeout}s; default action is REJECT."
        ) from None
      # Default approve — return checkpoint with APPROVE decision
      checkpoint.decision = HumanDecision.APPROVE
      checkpoint.feedback = f"Auto-approved after {self.timeout}s timeout."
      checkpoint.resolved_at = __import__("datetime").datetime.now(
        __import__("datetime").timezone.utc
      )
      return checkpoint

    response = response.strip().lower()

    if response in ("a", "approve", "y", "yes"):
      checkpoint.decision = HumanDecision.APPROVE
      checkpoint.feedback = "Approved by human."

    elif response in ("r", "reject", "n", "no"):
      checkpoint.decision = HumanDecision.REJECT
      checkpoint.feedback = "Rejected by human."
      checkpoint.resolved_at = __import__("datetime").datetime.now(
        __import__("datetime").timezone.utc
      )
      raise HumanRejectedError(
        f"Human rejected {checkpoint.trigger}: {checkpoint.description}"
      )

    elif response in ("m", "modify"):
      checkpoint.decision = HumanDecision.MODIFY
      checkpoint.feedback = "Modified by human."
      print("\nEnter modified arguments as JSON (press Enter twice to finish):")
      lines: list[str] = []
      while True:
        line = await asyncio.wait_for(self._read_input(), timeout=self.timeout)
        if line.strip() == "":
          break
        lines.append(line)
      try:
        import json

        modified = json.loads("\n".join(lines))
      except json.JSONDecodeError as exc:
        print(f"Invalid JSON: {exc}. Using original arguments.")
        modified = checkpoint.original_args
      checkpoint.modified_args = modified

    else:
      # Unknown response — treat as reject for safety
      checkpoint.decision = HumanDecision.REJECT
      checkpoint.feedback = f"Unknown response '{response}' treated as reject."
      checkpoint.resolved_at = __import__("datetime").datetime.now(
        __import__("datetime").timezone.utc
      )
      raise HumanRejectedError(
        f"Human rejected {checkpoint.trigger}: {checkpoint.description}"
      )

    checkpoint.resolved_at = __import__("datetime").datetime.now(
      __import__("datetime").timezone.utc
    )
    return checkpoint

  async def _read_input(self) -> str:
    """Async-friendly stdin read."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, input, "> ")


class MockApprover(HumanApprover):
  """Programmable approver for testing.

  Pre-configured with a sequence of decisions so tests can run without
  blocking on human input.

  Args:
    decisions: List of ``(decision, modified_args_or_none)`` tuples.
      ``decision`` is one of ``"approve"``, ``"modify"``, ``"reject"``.
    cycle: If ``True``, loop back to the start when decisions are exhausted.
  """

  def __init__(
    self,
    decisions: list[tuple[str, dict[str, Any] | None]] | None = None,
    *,
    cycle: bool = False,
  ):
    self._decisions = decisions or []
    self._index = 0
    self.cycle = cycle

  async def request_approval(self, checkpoint: HumanCheckpoint) -> HumanCheckpoint:
    if not self._decisions:
      raise RuntimeError("MockApprover has no pre-configured decisions.")

    idx = self._index % len(self._decisions) if self.cycle else self._index
    if idx >= len(self._decisions):
      raise RuntimeError(
        f"MockApprover ran out of decisions (index={self._index}, len={len(self._decisions)})"
      )

    decision_str, modified = self._decisions[idx]
    self._index += 1

    if decision_str == "approve":
      checkpoint.decision = HumanDecision.APPROVE
      checkpoint.feedback = "Auto-approved by MockApprover."

    elif decision_str == "reject":
      checkpoint.decision = HumanDecision.REJECT
      checkpoint.feedback = "Auto-rejected by MockApprover."
      checkpoint.resolved_at = __import__("datetime").datetime.now(
        __import__("datetime").timezone.utc
      )
      raise HumanRejectedError(
        f"MockApprover rejected {checkpoint.trigger}: {checkpoint.description}"
      )

    elif decision_str == "modify":
      checkpoint.decision = HumanDecision.MODIFY
      checkpoint.feedback = "Auto-modified by MockApprover."
      checkpoint.modified_args = modified or checkpoint.original_args

    else:
      raise ValueError(f"Unknown decision: {decision_str}")

    checkpoint.resolved_at = __import__("datetime").datetime.now(
      __import__("datetime").timezone.utc
    )
    return checkpoint
