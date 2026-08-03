from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from nonoka.core.agent import Agent
from nonoka.core.context import RunContext
from nonoka.core.memory import MemoryRole
from nonoka.core.runner import Runner
from nonoka.core.session import SessionStatus
from nonoka.core.tool import tool


def _make_runner(content: str = "final answer") -> Runner:
  """Create a Runner whose LLM always returns a plain final answer."""
  runner = Runner(checkpoint="memory", memory="in_memory")
  provider = MagicMock()
  provider.chat = AsyncMock(
    return_value=MagicMock(content=content, tool_calls=None, usage={})
  )
  provider.chat_stream = AsyncMock(return_value=iter([]))
  runner._create_llm = lambda agent: provider  # type: ignore[method-assign]
  runner.llm = provider
  return runner


def _counting_tool() -> tuple[Any, dict[str, int]]:
  """A tool that records how many times it was invoked."""
  calls = {"count": 0}

  @tool
  async def probe(ctx: RunContext) -> str:
    calls["count"] += 1
    return "probe result"

  return probe, calls


def _tool_call(call_id: str, name: str = "probe") -> dict[str, Any]:
  return {
    "id": call_id,
    "type": "function",
    "function": {"name": name, "arguments": "{}"},
  }


@pytest.mark.asyncio
async def test_resume_replays_dangling_tool_calls():
  """A crash after the assistant tool_calls message is repaired on resume."""
  probe, calls = _counting_tool()
  agent = Agent(model="test", tools=[probe])
  runner = _make_runner()

  # Simulate a crash: assistant tool_calls persisted, tool result never written.
  session = await runner._create_session(agent, deps=None)
  await session.memory.add("run the probe", MemoryRole.USER)
  await session.memory.add(
    "",
    MemoryRole.ASSISTANT,
    tool_calls=[_tool_call("call_1")],
  )
  session.status = SessionStatus.PAUSED
  await runner.checkpoint_store.save_session(session.session_id, session.to_state())

  result = await runner.resume(agent, session_id=session.session_id, deps=None)

  assert calls["count"] == 1
  assert result.success is True
  assert result.data == "final answer"
  tool_entries = [
    e for e in result.session.memory.entries if e.role == MemoryRole.TOOL
  ]
  assert len(tool_entries) == 1
  assert tool_entries[0].metadata["tool_call_id"] == "call_1"
  assert tool_entries[0].metadata["tool_name"] == "probe"
  assert "probe result" in tool_entries[0].content


@pytest.mark.asyncio
async def test_resume_replays_only_unanswered_tool_calls():
  """Calls that already have a TOOL result are not re-executed."""
  probe, calls = _counting_tool()
  agent = Agent(model="test", tools=[probe])
  runner = _make_runner()

  session = await runner._create_session(agent, deps=None)
  await session.memory.add("run two probes", MemoryRole.USER)
  await session.memory.add(
    "",
    MemoryRole.ASSISTANT,
    tool_calls=[_tool_call("call_1"), _tool_call("call_2")],
  )
  await session.memory.add(
    "probe result",
    MemoryRole.TOOL,
    tool_call_id="call_1",
    tool_name="probe",
  )
  session.status = SessionStatus.PAUSED
  await runner.checkpoint_store.save_session(session.session_id, session.to_state())

  result = await runner.resume(agent, session_id=session.session_id, deps=None)

  # Only call_2 was dangling, so the tool runs exactly once.
  assert calls["count"] == 1
  assert result.success is True
  tool_entries = [
    e for e in result.session.memory.entries if e.role == MemoryRole.TOOL
  ]
  assert {e.metadata["tool_call_id"] for e in tool_entries} == {"call_1", "call_2"}


@pytest.mark.asyncio
async def test_resume_without_dangling_tool_calls_behaves_as_before():
  """Without dangling calls, resume() does not execute any tool."""
  probe, calls = _counting_tool()
  agent = Agent(model="test", tools=[probe])
  runner = _make_runner()

  session = await runner._create_session(agent, deps=None)
  await session.memory.add("hello", MemoryRole.USER)
  await session.memory.add("hi", MemoryRole.ASSISTANT)
  session.status = SessionStatus.PAUSED
  await runner.checkpoint_store.save_session(session.session_id, session.to_state())

  result = await runner.resume(agent, session_id=session.session_id, deps=None)

  assert calls["count"] == 0
  assert result.success is True
  assert result.data == "final answer"
