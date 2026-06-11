from __future__ import annotations

from enum import Enum
from typing import Any


class AgentError(Exception):
  """Nonoka Agent base error"""
  pass


class CancelledError(AgentError):
  """Execution was cancelled by external request (e.g. Ctrl+C, timeout)."""
  pass


class TransientError(AgentError):
  """Temporary error (such as network timeout, interface rate limiting, etc.), suitable for retry"""
  pass


class SchemaError(AgentError):
  """Parameter / return value format does not match, usually requires LLM to regenerate"""
  pass


class LogicError(AgentError):
  """Tool internal irrecoverable logic error"""
  pass


class SafetyError(AgentError):
  """Safety check or authentication failure"""
  pass


class ResourceError(AgentError):
  """Resource exhaustion (such as Token exceeded)"""
  pass


class MaxTurnsExceeded(AgentError):
  """max turns exceeded"""
  pass


class MaxStepsExceeded(AgentError):
  """max steps exceeded"""
  pass


class HumanRejectedError(SafetyError):
  """Human-in-the-loop rejected the operation.

  Raised when a human approver explicitly rejects a tool call or plan step.
  By default the ``ErrorPolicy`` maps this to ``HALT`` so the run terminates
  rather than feeding the rejection back to the LLM as an observation.
  """
  pass


class ApprovalTimeoutError(SafetyError):
  """Human-in-the-loop approval request timed out.

  Raised when an approver does not respond within the configured deadline.
  The default disposition is ``HALT``.
  """
  pass


class ToolFatalError(AgentError):
  """Tool execution failed with a fatal error that should terminate the run.

  Raised when ErrorPolicy decides ``FAIL`` or ``HALT``.  The ReAct loop
  catches this (but not generic exceptions) to produce a terminal
  ``RunResult`` instead of feeding the error back to the LLM as an
  observation.
  """
  pass


class ToolErrorActionType(str, Enum):
  RETRY = "retry"
  HALT = "halt"
  REPORT = "report"
  FAIL = "fail"


class ToolErrorAction:
  """Describe the next action for the scheduler to handle tool errors"""
  def __init__(self, action_type: ToolErrorActionType, **kwargs: Any):
    self.type = action_type
    self.kwargs = kwargs

  @classmethod
  def retry(cls, max_retries: int = 3):
    return cls(ToolErrorActionType.RETRY, max_retries=max_retries)

  @classmethod
  def halt(cls, require_approval: bool = True):
    return cls(ToolErrorActionType.HALT, require_approval=require_approval)

  @classmethod
  def report_to_llm(cls):
    """Report the error information as an Observation to the LLM"""
    return cls(ToolErrorActionType.REPORT)

  @classmethod
  def fail(cls):
    return cls(ToolErrorActionType.FAIL)


class ErrorPolicy:
  """Default user-configurable error handling policy"""

  def on_tool_error(self, error: Exception, step_id: str) -> ToolErrorAction:
    """Give suggested response strategy based on exception type"""
    if isinstance(error, TransientError):
      return ToolErrorAction.retry(max_retries=3)
    elif isinstance(error, SafetyError):
      return ToolErrorAction.halt(require_approval=True)
    elif isinstance(error, LogicError) or isinstance(error, SchemaError):
      return ToolErrorAction.report_to_llm()
    else:
      return ToolErrorAction.fail()
