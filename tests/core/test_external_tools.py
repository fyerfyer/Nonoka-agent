"""Tests for external tool execution delegation."""

from __future__ import annotations

from typing import Any

import pytest

from nonoka import ExternalCapability, Runner
from nonoka.core.agent import Agent
from nonoka.core.context import RunContext
from nonoka.core.errors import ExternalToolExecutionRequiredError
from nonoka.core.memory import MemoryRole
from nonoka.core.paradigm import ReActAgent
from nonoka.core.runner import StreamEvent


class _ExternalCapability:
  """A capability whose execution is delegated to an external host."""

  external = True

  def __init__(self, name: str, description: str, parameters: dict[str, Any]):
    self.name = name
    self.description = description
    self.parameters = parameters

  async def invoke(self, ctx: RunContext, arguments: dict[str, Any]) -> Any:
    raise RuntimeError("External tools should not be invoked inside nonoka.")

  def to_json_schema(self) -> dict[str, Any]:
    return {
      "type": "function",
      "function": {
        "name": self.name,
        "description": self.description,
        "parameters": self.parameters,
      },
    }


class _LocalCapability:
  """A normal capability that executes inside nonoka."""

  def __init__(self):
    self.name = "local_echo"
    self.description = "Echoes the input."
    self.parameters = {
      "type": "object",
      "properties": {"value": {"type": "string"}},
      "required": ["value"],
    }

  async def invoke(self, ctx: RunContext, arguments: dict[str, Any]) -> Any:
    return {"echo": arguments.get("value")}

  def to_json_schema(self) -> dict[str, Any]:
    return {
      "type": "function",
      "function": {
        "name": self.name,
        "description": self.description,
        "parameters": self.parameters,
      },
    }


def test_external_capability_has_external_marker():
  cap = _ExternalCapability("bash", "Run shell commands", {"type": "object", "properties": {}})
  assert cap.external is True


def test_external_capability_metadata_not_in_schema():
  cap = ExternalCapability(
    name="bash",
    description="Run shell commands",
    parameters={"type": "object", "properties": {}},
    metadata={"kind": "host_tool", "original_name": "bash"},
  )
  assert cap.metadata == {"kind": "host_tool", "original_name": "bash"}
  schema = cap.to_json_schema()
  assert schema["function"]["name"] == "bash"
  assert "metadata" not in schema["function"]


def test_external_capability_to_json_schema():
  cap = _ExternalCapability("bash", "Run shell commands", {"type": "object", "properties": {}})
  schema = cap.to_json_schema()
  assert schema["function"]["name"] == "bash"
  assert schema["function"]["description"] == "Run shell commands"


@pytest.mark.asyncio
async def test_execute_tool_call_raises_external_tool_execution_required():
  agent = Agent(
    model="gpt-4o",
    tools=[_ExternalCapability("bash", "Run commands", {"type": "object", "properties": {}})],
    system_prompt="You are a test agent.",
  )
  runner = Runner(checkpoint="memory", memory="disabled")

  # Manually trigger the ReAct loop with a session that already has a pending
  # assistant message with tool_calls, so we hit _execute_tool_call directly.
  session = await runner._create_session(agent, deps=None)
  await session.memory.add("run ls", MemoryRole.USER)
  await session.memory.add(
    "",
    MemoryRole.ASSISTANT,
    tool_calls=[
      {
        "id": "call_1",
        "type": "function",
        "function": {"name": "bash", "arguments": '{"command": "ls"}'},
      }
    ],
  )

  paradigm = ReActAgent()
  with pytest.raises(ExternalToolExecutionRequiredError) as exc_info:
    await paradigm._execute_tool_call(
      session,
      runner,
      {
        "id": "call_1",
        "type": "function",
        "function": {"name": "bash", "arguments": '{"command": "ls"}'},
      },
    )

  assert exc_info.value.tool_name == "bash"
  assert exc_info.value.tool_call_id == "call_1"
  assert exc_info.value.arguments == {"command": "ls"}


@pytest.mark.asyncio
async def test_run_stream_pauses_on_external_tool():
  """When the only tool is external, the stream pauses after tool_call_start."""
  agent = Agent(
    model="gpt-4o",
    tools=[_ExternalCapability("bash", "Run commands", {"type": "object", "properties": {}})],
    system_prompt="You are a test agent.",
  )
  runner = Runner(checkpoint="memory", memory="disabled")

  # Mock the LLM to force a tool call.
  class _FakeLLM:
    async def chat_stream(self, messages, tools=None):
      yield type("Chunk", (), {"content_delta": "", "tool_call_deltas": None, "finish_reason": None})()
      yield type(
        "Chunk",
        (),
        {
          "content_delta": "",
          "tool_call_deltas": [
            {"index": 0, "id": "call_1", "function": {"name": "bash", "arguments": '{"command": "ls"}'}}
          ],
          "finish_reason": "tool_calls",
        },
      )()

  runner._llm_cache["gpt-4o"] = _FakeLLM()
  runner.llm = _FakeLLM()

  events: list[StreamEvent] = []
  async for event in runner.run_react_stream(agent, "run ls", deps=None):
    events.append(event)

  types = [e.type for e in events]
  assert "tool_call_start" in types
  assert "final" in types

  final = next(e for e in events if e.type == "final")
  assert final.data.get("requires_external_execution") is True


@pytest.mark.asyncio
async def test_resume_external_tools_injects_result_and_continues():
  """After pausing for an external tool, resume injects the result."""
  agent = Agent(
    model="gpt-4o",
    tools=[_ExternalCapability("bash", "Run commands", {"type": "object", "properties": {}})],
    system_prompt="You are a test agent.",
  )
  runner = Runner(checkpoint="memory", memory="disabled")

  # Seed a paused session.
  session = await runner._create_session(agent, deps=None, session_id="sess-1")
  await session.memory.add("run ls", MemoryRole.USER)
  await session.memory.add(
    "",
    MemoryRole.ASSISTANT,
    tool_calls=[
      {
        "id": "call_1",
        "type": "function",
        "function": {"name": "bash", "arguments": '{"command": "ls"}'},
      }
    ],
  )
  session.status = "paused"
  await runner.checkpoint_store.save_session("sess-1", session.to_state())

  # Mock the LLM final response after the tool result is injected.
  class _FakeLLM:
    async def chat_stream(self, messages, tools=None):
      # The resumed session should see a user msg, assistant tool_call, and tool result.
      yield type("Chunk", (), {"content_delta": "done", "tool_call_deltas": None, "finish_reason": "stop"})()

  runner._llm_cache["gpt-4o"] = _FakeLLM()
  runner.llm = _FakeLLM()

  events: list[StreamEvent] = []
  async for event in runner.resume_external_tools(agent, deps=None, session_id="sess-1", results={"call_1": "file.txt"}):
    events.append(event)

  # After resume, the tool result should be in the checkpoint memory and the
  # loop should complete. The runner created a new session object, so load it.
  resumed_state = await runner.checkpoint_store.load_session("sess-1")
  assert resumed_state is not None
  tool_entries = [e for e in resumed_state.memory_entries if e["role"] == MemoryRole.TOOL]
  assert len(tool_entries) == 1
  assert "file.txt" in tool_entries[0]["content"]

  final = next((e for e in events if e.type == "final"), None)
  assert final is not None
  assert final.data.get("success") is True
