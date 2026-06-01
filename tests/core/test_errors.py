import pytest
from nonoka.core.errors import (
  ErrorPolicy,
  TransientError,
  SafetyError,
  LogicError,
  ToolErrorActionType
)


def test_error_policy_routing():
  policy = ErrorPolicy()

  action = policy.on_tool_error(TransientError("timeout"), "step-1")
  assert action.type == ToolErrorActionType.RETRY
  assert action.kwargs.get("max_retries") == 3

  action = policy.on_tool_error(SafetyError("blocked"), "step-2")
  assert action.type == ToolErrorActionType.HALT
  assert action.kwargs.get("require_approval") is True

  action = policy.on_tool_error(LogicError("file not found"), "step-3")
  assert action.type == ToolErrorActionType.REPORT
