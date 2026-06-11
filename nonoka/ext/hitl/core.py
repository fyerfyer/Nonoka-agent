from __future__ import annotations

import uuid
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from nonoka.core.errors import HumanRejectedError, ApprovalTimeoutError


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #

class HumanDecision(str, Enum):
  """Possible human decisions on a checkpoint."""

  APPROVE = "approve"
  MODIFY = "modify"
  REJECT = "reject"


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #

@dataclass
class ToolRule:
  """Rule that decides whether a tool call requires human approval.

  Args:
    tool: Name of the tool to match.  Use ``"*"`` to match all tools.
    pattern: Optional regex pattern applied to the JSON-serialised arguments.
      If provided, both *tool* AND *pattern* must match for the rule to fire.
    action: ``"approve"`` (approve/reject only) or ``"modify"`` (can edit args).
    description: Optional human-readable description shown in the checkpoint.
  """

  tool: str
  pattern: str | None = None
  action: str = "approve"  # "approve" or "modify"
  description: str = ""

  def matches(self, tool_name: str, arguments: dict[str, Any]) -> bool:
    """Check whether this rule matches the given tool call."""
    # Tool name match: exact or wildcard
    if self.tool != "*" and self.tool != tool_name:
      return False

    # Pattern match (optional)
    if self.pattern is not None:
      args_text = _serialise_args(arguments)
      if not re.search(self.pattern, args_text, re.IGNORECASE):
        return False

    return True


def _serialise_args(arguments: dict[str, Any]) -> str:
  """Serialise arguments to a string for pattern matching."""
  import json

  try:
    return json.dumps(arguments, ensure_ascii=False, default=str)
  except (TypeError, ValueError):
    return str(arguments)


@dataclass
class HumanCheckpoint:
  """A checkpoint created when human approval is required.

  The approver receives a checkpoint, presents it to the human, and
  returns a **mutated** checkpoint with ``decision``, ``feedback``,
  and optionally ``modified_args`` filled in.
  """

  checkpoint_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
  trigger: str = ""  # e.g. "tool_call:edit_file"
  description: str = ""  # Human-readable summary
  context: dict[str, Any] = field(default_factory=dict)  # session_id, turn_count, etc.
  original_args: dict[str, Any] = field(default_factory=dict)
  decision: HumanDecision | None = None
  feedback: str = ""
  modified_args: dict[str, Any] | None = None
  created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
  resolved_at: datetime | None = None

  @property
  def is_resolved(self) -> bool:
    """Whether a decision has been recorded."""
    return self.decision is not None

  @property
  def effective_args(self) -> dict[str, Any]:
    """The arguments that should be used for execution.

    Returns ``modified_args`` when the decision is ``MODIFY``,
    otherwise returns ``original_args``.
    """
    if self.decision == HumanDecision.MODIFY and self.modified_args is not None:
      return self.modified_args
    return self.original_args


# --------------------------------------------------------------------------- #
# HumanApprover Protocol
# --------------------------------------------------------------------------- #

@runtime_checkable
class HumanApprover(Protocol):
  """Protocol for human approval backends.

  Implementations are responsible for presenting the checkpoint to a human
  and returning their decision.  The protocol is intentionally minimal so
  that CLI, WebSocket, Slack Bot, email, etc. can all be supported.
  """

  async def request_approval(self, checkpoint: HumanCheckpoint) -> HumanCheckpoint:
    """Present *checkpoint* to a human and return the resolved checkpoint.

    The returned checkpoint must have ``decision`` set.  It may also have
    ``modified_args`` populated (when decision is ``MODIFY``).

    Raises:
      HumanRejectedError: when the human explicitly rejects.
      ApprovalTimeoutError: when the request times out.
    """
    ...


# --------------------------------------------------------------------------- #
# Re-export errors so users can import everything from nonoka.ext.hitl
# --------------------------------------------------------------------------- #

__all__ = [
  "HumanDecision",
  "ToolRule",
  "HumanCheckpoint",
  "HumanApprover",
  "HumanRejectedError",
  "ApprovalTimeoutError",
]
