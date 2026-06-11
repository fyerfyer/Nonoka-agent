from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from nonoka.core.agent import Agent
from nonoka.core.context import RunContext
from nonoka.core.hooks import Hooks, HookContext
from nonoka.core.paradigm import ReActAgent, PlanExecutor
from nonoka.core.runner import Runner
from nonoka.core.session import Session
from nonoka.core.tool import tool
from nonoka.core.types import RunResult
from nonoka.core.plan import PlanBuilder
from nonoka.core.errors import HumanRejectedError, ApprovalTimeoutError, SafetyError

from nonoka.ext.hitl.core import ToolRule, HumanCheckpoint, HumanDecision
from nonoka.ext.hitl.hooks import HumanInTheLoopHooks
from nonoka.ext.hitl.approvers import MockApprover


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _make_mock_runner(response_content: str = "done", tool_calls=None):
  runner = Runner(checkpoint="memory")
  provider = MagicMock()
  provider.chat = AsyncMock(
    return_value=MagicMock(
      content=response_content,
      tool_calls=tool_calls,
      usage={},
    )
  )
  provider.chat_stream = AsyncMock(return_value=iter([]))
  runner._create_llm = lambda agent: provider  # type: ignore[method-assign]
  runner.llm = provider
  return runner


# --------------------------------------------------------------------------- #
# 1. ToolRule matching
# --------------------------------------------------------------------------- #

def test_tool_rule_matches_exact_tool_name():
  rule = ToolRule(tool="edit_file", action="approve")
  assert rule.matches("edit_file", {"path": "/tmp/test"}) is True
  assert rule.matches("delete_file", {"path": "/tmp/test"}) is False


def test_tool_rule_wildcard_matches_all_tools():
  rule = ToolRule(tool="*", action="approve")
  assert rule.matches("edit_file", {}) is True
  assert rule.matches("run_command", {}) is True
  assert rule.matches("anything", {}) is True


def test_tool_rule_pattern_matches_arguments():
  rule = ToolRule(tool="run_command", pattern="rm|drop|delete", action="approve")
  assert rule.matches("run_command", {"cmd": "rm -rf /tmp"}) is True
  assert rule.matches("run_command", {"cmd": "ls -la"}) is False


def test_tool_rule_pattern_case_insensitive():
  rule = ToolRule(tool="run_command", pattern="rm", action="approve")
  assert rule.matches("run_command", {"cmd": "RM -rf /tmp"}) is True


def test_tool_rule_both_conditions_must_match():
  rule = ToolRule(tool="edit_file", pattern="secret", action="approve")
  assert rule.matches("edit_file", {"path": "/secret.txt"}) is True
  assert rule.matches("edit_file", {"path": "/public.txt"}) is False
  assert rule.matches("delete_file", {"path": "/secret.txt"}) is False


# --------------------------------------------------------------------------- #
# 2. HumanCheckpoint
# --------------------------------------------------------------------------- #

def test_checkpoint_is_resolved():
  cp = HumanCheckpoint(decision=HumanDecision.APPROVE)
  assert cp.is_resolved is True

  cp2 = HumanCheckpoint()
  assert cp2.is_resolved is False


def test_checkpoint_effective_args_approve():
  cp = HumanCheckpoint(
    original_args={"value": 1},
    decision=HumanDecision.APPROVE,
  )
  assert cp.effective_args == {"value": 1}


def test_checkpoint_effective_args_modify():
  cp = HumanCheckpoint(
    original_args={"value": 1},
    decision=HumanDecision.MODIFY,
    modified_args={"value": 999},
  )
  assert cp.effective_args == {"value": 999}


def test_checkpoint_effective_args_modify_without_modified_args():
  cp = HumanCheckpoint(
    original_args={"value": 1},
    decision=HumanDecision.MODIFY,
    modified_args=None,
  )
  assert cp.effective_args == {"value": 1}


# --------------------------------------------------------------------------- #
# 3. MockApprover
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_mock_approver_approve():
  approver = MockApprover(decisions=[("approve", None)])
  cp = HumanCheckpoint(trigger="tool_call:test")

  result = await approver.request_approval(cp)

  assert result.decision == HumanDecision.APPROVE
  assert result.is_resolved is True


@pytest.mark.asyncio
async def test_mock_approver_reject():
  approver = MockApprover(decisions=[("reject", None)])
  cp = HumanCheckpoint(trigger="tool_call:test")

  with pytest.raises(HumanRejectedError):
    await approver.request_approval(cp)


@pytest.mark.asyncio
async def test_mock_approver_modify():
  approver = MockApprover(decisions=[("modify", {"value": 42})])
  cp = HumanCheckpoint(trigger="tool_call:test", original_args={"value": 1})

  result = await approver.request_approval(cp)

  assert result.decision == HumanDecision.MODIFY
  assert result.effective_args == {"value": 42}


@pytest.mark.asyncio
async def test_mock_approver_sequence():
  approver = MockApprover(
    decisions=[("approve", None), ("reject", None), ("modify", {"x": 2})]
  )

  cp1 = await approver.request_approval(HumanCheckpoint())
  assert cp1.decision == HumanDecision.APPROVE

  with pytest.raises(HumanRejectedError):
    await approver.request_approval(HumanCheckpoint())

  cp3 = await approver.request_approval(HumanCheckpoint(original_args={"x": 1}))
  assert cp3.decision == HumanDecision.MODIFY
  assert cp3.effective_args == {"x": 2}


@pytest.mark.asyncio
async def test_mock_approver_exhaustion():
  approver = MockApprover(decisions=[("approve", None)])

  await approver.request_approval(HumanCheckpoint())

  with pytest.raises(RuntimeError, match="ran out of decisions"):
    await approver.request_approval(HumanCheckpoint())


@pytest.mark.asyncio
async def test_mock_approver_cycle():
  approver = MockApprover(decisions=[("approve", None)], cycle=True)

  for _ in range(5):
    cp = await approver.request_approval(HumanCheckpoint())
    assert cp.decision == HumanDecision.APPROVE


# --------------------------------------------------------------------------- #
# 4. HumanInTheLoopHooks — intercept logic
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_hitl_no_rule_match_returns_original_args():
  """When no rule matches, arguments should pass through unchanged."""
  approver = MockApprover(decisions=[])
  hitl = HumanInTheLoopHooks(
    approver=approver,
    rules=[ToolRule(tool="edit_file", action="approve")],
  )

  agent = Agent(model="test", tools=[])
  session = Session(session_id="s1", agent=agent, deps=None)
  ctx = HookContext(session=session, runner=MagicMock())

  result = await hitl.on_tool_start_intercept(ctx, "read_file", {"path": "/tmp"})

  assert result == {"path": "/tmp"}


@pytest.mark.asyncio
async def test_hitl_rule_match_triggers_approval():
  """When a rule matches, the approver should be invoked."""
  approver = MockApprover(decisions=[("approve", None)])
  hitl = HumanInTheLoopHooks(
    approver=approver,
    rules=[ToolRule(tool="edit_file", action="approve")],
  )

  agent = Agent(model="test", tools=[])
  session = Session(session_id="s1", agent=agent, deps=None)
  ctx = HookContext(session=session, runner=MagicMock())

  result = await hitl.on_tool_start_intercept(ctx, "edit_file", {"path": "/tmp"})

  assert result == {"path": "/tmp"}
  assert approver._index == 1


@pytest.mark.asyncio
async def test_hitl_rule_match_reject_raises():
  """When the human rejects, HumanRejectedError should be raised."""
  approver = MockApprover(decisions=[("reject", None)])
  hitl = HumanInTheLoopHooks(
    approver=approver,
    rules=[ToolRule(tool="delete_file", action="approve")],
  )

  agent = Agent(model="test", tools=[])
  session = Session(session_id="s1", agent=agent, deps=None)
  ctx = HookContext(session=session, runner=MagicMock())

  with pytest.raises(HumanRejectedError):
    await hitl.on_tool_start_intercept(ctx, "delete_file", {"path": "/tmp"})


@pytest.mark.asyncio
async def test_hitl_rule_match_modifies_args():
  """When the human modifies args, the modified args should be returned."""
  approver = MockApprover(decisions=[("modify", {"path": "/safe/path"})])
  hitl = HumanInTheLoopHooks(
    approver=approver,
    rules=[ToolRule(tool="edit_file", action="modify")],
  )

  agent = Agent(model="test", tools=[])
  session = Session(session_id="s1", agent=agent, deps=None)
  ctx = HookContext(session=session, runner=MagicMock())

  result = await hitl.on_tool_start_intercept(ctx, "edit_file", {"path": "/dangerous"})

  assert result == {"path": "/safe/path"}


@pytest.mark.asyncio
async def test_hitl_default_action_approve():
  """When default_action='approve', every tool call should trigger approval
  even without a matching rule."""
  approver = MockApprover(decisions=[("approve", None)])
  hitl = HumanInTheLoopHooks(
    approver=approver,
    rules=[],
    default_action="approve",
  )

  agent = Agent(model="test", tools=[])
  session = Session(session_id="s1", agent=agent, deps=None)
  ctx = HookContext(session=session, runner=MagicMock())

  result = await hitl.on_tool_start_intercept(ctx, "any_tool", {"x": 1})

  assert result == {"x": 1}
  assert approver._index == 1


# --------------------------------------------------------------------------- #
# 5. Integration with ReActAgent (mock LLM)
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_hitl_blocks_tool_call_in_react():
  """When HITL rejects a tool call inside ReAct, the error should propagate
  and terminate execution (since HumanRejectedError is a SafetyError -> HALT)."""

  @tool
  async def dangerous_delete(ctx: RunContext, path: str) -> str:
    return f"deleted {path}"

  hitl = HumanInTheLoopHooks(
    approver=MockApprover(decisions=[("reject", None)]),
    rules=[ToolRule(tool="dangerous_delete", action="approve")],
  )

  runner = Runner(checkpoint="memory", hooks=hitl)
  provider = MagicMock()
  provider.chat = AsyncMock(
    return_value=MagicMock(
      content="",
      tool_calls=[{
        "id": "call_1",
        "type": "function",
        "function": {"name": "dangerous_delete", "arguments": '{"path": "/etc/passwd"}'},
      }],
      usage={},
    )
  )
  runner._create_llm = lambda agent: provider  # type: ignore[method-assign]
  runner.llm = provider

  agent = Agent(model="test", tools=[dangerous_delete], max_turns=3)

  result = await runner.run_react(agent, prompt="delete something", deps=None)

  assert result.success is False
  assert result.error_type == "halted"
  assert "rejected" in result.error.lower() or "Human" in result.error


@pytest.mark.asyncio
async def test_hitl_modifies_tool_args_in_react():
  """When HITL modifies tool args, the modified args should be passed to
  the actual tool function."""

  received_args = {}

  @tool
  async def write_file(ctx: RunContext, path: str, content: str) -> str:
    received_args["path"] = path
    received_args["content"] = content
    return f"wrote to {path}"

  hitl = HumanInTheLoopHooks(
    approver=MockApprover(decisions=[("modify", {"path": "/safe/file.txt", "content": "safe"})]),
    rules=[ToolRule(tool="write_file", action="modify")],
  )

  runner = Runner(checkpoint="memory", hooks=hitl)
  provider = MagicMock()
  provider.chat = AsyncMock(
    return_value=MagicMock(
      content="done",
      tool_calls=[{
        "id": "call_1",
        "type": "function",
        "function": {
          "name": "write_file",
          "arguments": '{"path": "/etc/passwd", "content": "hacked"}',
        },
      }],
      usage={},
    )
  )
  # Second LLM call — final answer
  provider.chat = AsyncMock(side_effect=[
    MagicMock(
      content="",
      tool_calls=[{
        "id": "call_1",
        "type": "function",
        "function": {
          "name": "write_file",
          "arguments": '{"path": "/etc/passwd", "content": "hacked"}',
        },
      }],
      usage={},
    ),
    MagicMock(content="Done", tool_calls=None, usage={}),
  ])
  runner._create_llm = lambda agent: provider  # type: ignore[method-assign]
  runner.llm = provider

  agent = Agent(model="test", tools=[write_file], max_turns=3)

  result = await runner.run_react(agent, prompt="write something", deps=None)

  # The tool should have received MODIFIED args, not the original dangerous ones
  assert received_args["path"] == "/safe/file.txt"
  assert received_args["content"] == "safe"


@pytest.mark.asyncio
async def test_hitl_allows_tool_call_in_react():
  """When HITL approves a tool call, execution should proceed normally."""

  @tool
  async def safe_read(ctx: RunContext, path: str) -> str:
    return f"content of {path}"

  hitl = HumanInTheLoopHooks(
    approver=MockApprover(decisions=[("approve", None)]),
    rules=[ToolRule(tool="safe_read", action="approve")],
  )

  runner = Runner(checkpoint="memory", hooks=hitl)
  provider = MagicMock()
  provider.chat = AsyncMock(side_effect=[
    MagicMock(
      content="",
      tool_calls=[{
        "id": "call_1",
        "type": "function",
        "function": {"name": "safe_read", "arguments": '{"path": "/tmp/test"}'},
      }],
      usage={},
    ),
    MagicMock(content="Here is the content: hello", tool_calls=None, usage={}),
  ])
  runner._create_llm = lambda agent: provider  # type: ignore[method-assign]
  runner.llm = provider

  agent = Agent(model="test", tools=[safe_read], max_turns=3)

  result = await runner.run_react(agent, prompt="read file", deps=None)

  assert result.success is True
  assert "hello" in str(result.data)


# --------------------------------------------------------------------------- #
# 6. Integration with PlanExecutor
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_hitl_blocks_plan_step():
  """When HITL rejects a plan step, the plan should fail with halted error."""

  @tool
  async def delete_resource(ctx: RunContext, name: str) -> str:
    return f"deleted {name}"

  hitl = HumanInTheLoopHooks(
    approver=MockApprover(decisions=[("reject", None)]),
    rules=[ToolRule(tool="delete_resource", action="approve")],
  )

  runner = Runner(checkpoint="memory", hooks=hitl)
  runner._create_llm = lambda agent: MagicMock()  # type: ignore[method-assign]
  runner.llm = MagicMock()

  agent = Agent(model="test", tools=[delete_resource])
  plan = PlanBuilder(objective="Clean up").step("s1", delete_resource, name="prod-db").build()

  result = await runner.run_plan(agent, plan=plan, deps=None)

  assert result.success is False
  assert "rejected" in result.error.lower() or "Human" in result.error


@pytest.mark.asyncio
async def test_hitl_modifies_plan_step_args():
  """When HITL modifies plan step args, the tool should receive modified args."""

  received_name = None

  @tool
  async def deploy_app(ctx: RunContext, name: str) -> str:
    nonlocal received_name
    received_name = name
    return f"deployed {name}"

  hitl = HumanInTheLoopHooks(
    approver=MockApprover(decisions=[("modify", {"name": "safe-app"})]),
    rules=[ToolRule(tool="deploy_app", action="modify")],
  )

  runner = Runner(checkpoint="memory", hooks=hitl)
  runner._create_llm = lambda agent: MagicMock()  # type: ignore[method-assign]
  runner.llm = MagicMock()

  agent = Agent(model="test", tools=[deploy_app])
  plan = PlanBuilder(objective="Deploy").step("s1", deploy_app, name="dangerous-app").build()

  result = await runner.run_plan(agent, plan=plan, deps=None)

  assert result.success is True
  assert received_name == "safe-app"


# --------------------------------------------------------------------------- #
# 7. Pattern-based rule matching in real execution
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_hitl_pattern_rule_blocks_dangerous_command():
  """A pattern rule should catch dangerous arguments even when the tool
  name itself is benign."""

  @tool
  async def run_command(ctx: RunContext, cmd: str) -> str:
    return f"ran: {cmd}"

  hitl = HumanInTheLoopHooks(
    approver=MockApprover(decisions=[("reject", None)]),
    rules=[ToolRule(tool="run_command", pattern="rm.*-rf|DROP TABLE", action="approve")],
  )

  runner = Runner(checkpoint="memory", hooks=hitl)
  provider = MagicMock()
  provider.chat = AsyncMock(
    return_value=MagicMock(
      content="",
      tool_calls=[{
        "id": "call_1",
        "type": "function",
        "function": {"name": "run_command", "arguments": '{"cmd": "rm -rf /"}'},
      }],
      usage={},
    )
  )
  runner._create_llm = lambda agent: provider  # type: ignore[method-assign]
  runner.llm = provider

  agent = Agent(model="test", tools=[run_command], max_turns=3)

  result = await runner.run_react(agent, prompt="run dangerous command", deps=None)

  assert result.success is False
  assert result.error_type == "halted"


@pytest.mark.asyncio
async def test_hitl_pattern_rule_allows_safe_command():
  """A pattern rule should NOT block safe arguments."""

  @tool
  async def run_command(ctx: RunContext, cmd: str) -> str:
    return f"ran: {cmd}"

  hitl = HumanInTheLoopHooks(
    approver=MockApprover(decisions=[]),  # Should never be called
    rules=[ToolRule(tool="run_command", pattern="rm.*-rf", action="approve")],
  )

  runner = Runner(checkpoint="memory", hooks=hitl)
  provider = MagicMock()
  provider.chat = AsyncMock(side_effect=[
    MagicMock(
      content="",
      tool_calls=[{
        "id": "call_1",
        "type": "function",
        "function": {"name": "run_command", "arguments": '{"cmd": "ls -la"}'},
      }],
      usage={},
    ),
    MagicMock(content="Output is here", tool_calls=None, usage={}),
  ])
  runner._create_llm = lambda agent: provider  # type: ignore[method-assign]
  runner.llm = provider

  agent = Agent(model="test", tools=[run_command], max_turns=3)

  result = await runner.run_react(agent, prompt="list files", deps=None)

  assert result.success is True


# --------------------------------------------------------------------------- #
# 8. HumanRejectedError is a SafetyError
# --------------------------------------------------------------------------- #

def test_human_rejected_error_is_safety_error():
  """HumanRejectedError should subclass SafetyError so ErrorPolicy maps it to HALT."""
  assert issubclass(HumanRejectedError, SafetyError)


def test_approval_timeout_error_is_safety_error():
  """ApprovalTimeoutError should subclass SafetyError so ErrorPolicy maps it to HALT."""
  assert issubclass(ApprovalTimeoutError, SafetyError)


# --------------------------------------------------------------------------- #
# 9. Multiple tool calls in a single turn
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_hitl_multiple_tool_calls_independent():
  """Each tool call in a turn should be independently evaluated by HITL."""

  @tool
  async def read_file(ctx: RunContext, path: str) -> str:
    return f"content of {path}"

  @tool
  async def delete_file(ctx: RunContext, path: str) -> str:
    return f"deleted {path}"

  # Approve read_file, reject delete_file
  hitl = HumanInTheLoopHooks(
    approver=MockApprover(decisions=[("approve", None), ("reject", None)]),
    rules=[
      ToolRule(tool="read_file", action="approve"),
      ToolRule(tool="delete_file", action="approve"),
    ],
  )

  runner = Runner(checkpoint="memory", hooks=hitl)
  provider = MagicMock()
  provider.chat = AsyncMock(
    return_value=MagicMock(
      content="",
      tool_calls=[
        {
          "id": "call_1",
          "type": "function",
          "function": {"name": "read_file", "arguments": '{"path": "/tmp/a"}'},
        },
        {
          "id": "call_2",
          "type": "function",
          "function": {"name": "delete_file", "arguments": '{"path": "/tmp/b"}'},
        },
      ],
      usage={},
    )
  )
  runner._create_llm = lambda agent: provider  # type: ignore[method-assign]
  runner.llm = provider

  agent = Agent(model="test", tools=[read_file, delete_file], max_turns=3)

  result = await runner.run_react(agent, prompt="read and delete", deps=None)

  # read_file should have been approved, delete_file rejected -> HALT
  assert result.success is False
  assert result.error_type == "halted"
