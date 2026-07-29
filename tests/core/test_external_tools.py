"""Tests for external tool execution delegation."""

from __future__ import annotations

from typing import Any

import pytest

from nonoka import (
  CompletionContract, EffectAttestation, ExternalCapability, ExternalToolReceipt,
  ObservationCompleteness, Runner, RuntimeLimits, ToolExecution,
  WorkspaceAttestation,
  VerificationKind, VerificationLevel, VerificationReceipt, VerificationStatus,
)
from nonoka.core.agent import Agent
from nonoka.core.context import RunContext
from nonoka.core.errors import ExternalToolExecutionRequiredError
from nonoka.core.memory import MemoryRole
from nonoka.core.paradigm import ReActAgent
from nonoka.core.runner import StreamEvent
from nonoka.core.session import Session
from nonoka.ext.coding import WorkspaceProgressExtension


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


def test_observation_fallback_guidance_uses_capability_metadata():
  local = _LocalCapability()
  local.name = "local_evidence"
  local.metadata = {"kind": "observation_fallback"}
  session = Session("fallback-guidance", Agent(model="gpt-4o", tools=[local]))

  guidance = ReActAgent._observation_fallback_guidance(session)

  assert "local_evidence" in guidance
  assert "partial" in guidance


@pytest.mark.asyncio
async def test_partial_external_receipt_runs_declared_read_only_fallback():
  """A generic declaration, rather than a host/tool-name special case, drives recovery."""
  class _MappedFallback(_LocalCapability):
    def __init__(self):
      super().__init__()
      self.name = "bounded_local_probe"
      self.execution = ToolExecution(read_only=True)
      self.metadata = {
        "kind": "observation_fallback",
        "fallback": {
          "on_partial_external": True,
          "argument_map": {"query": "needle", "scope": "root"},
          "defaults": {"strategy": "bounded"},
        },
      }

    async def invoke(self, ctx: RunContext, arguments: dict[str, Any]) -> Any:
      return {"received": arguments}

  external = _ExternalCapability(
    "unrelated_host_probe",
    "A host-defined observation tool.",
    {"type": "object"},
  )
  fallback = _MappedFallback()
  agent = Agent(model="gpt-4o", tools=[external, fallback])
  runner = Runner(checkpoint="memory", memory="disabled")
  session = await runner._create_session(agent, deps=None, session_id="mapped-fallback")
  await session.memory.add("inspect source", MemoryRole.USER)
  await session.memory.add(
    "",
    MemoryRole.ASSISTANT,
    tool_calls=[{
      "id": "host_call",
      "function": {
        "name": "unrelated_host_probe",
        "arguments": '{"needle":"secret","root":"payload"}',
      },
    }],
  )
  session.status = "paused"
  await runner.checkpoint_store.save_session(session.session_id, session.to_state())

  class _FinalLLM:
    async def chat_stream(self, messages, tools=None, **_kwargs):
      yield type("Chunk", (), {
        "content_delta": "bounded evidence considered",
        "tool_call_deltas": None,
        "finish_reason": "stop",
      })()

  runner._llm_cache["gpt-4o"] = _FinalLLM()
  events = []
  async for event in runner.resume_external_tools(
    agent,
    deps=None,
    session_id=session.session_id,
    results={"host_call": ExternalToolReceipt(
      result="host preview",
      completeness=ObservationCompleteness.PARTIAL,
    )},
  ):
    events.append(event)

  assert next(event for event in events if event.type == "final").data["success"] is True
  state = await runner.checkpoint_store.load_session(session.session_id)
  observation = next(entry for entry in state.memory_entries if entry["role"] == MemoryRole.TOOL)
  assert "Automatic local observation fallback: bounded_local_probe" in observation["content"]
  assert '"query": "secret"' in observation["content"]
  assert '"scope": "payload"' in observation["content"]
  assert '"strategy": "bounded"' in observation["content"]


@pytest.mark.asyncio
async def test_host_verification_receipt_satisfies_focused_completion_rule():
  external = _ExternalCapability(
    "bash", "Run commands", {"type": "object", "properties": {"command": {"type": "string"}}},
  )
  agent = Agent(
    model="gpt-4o",
    tools=[external],
    completion_contract=CompletionContract(require_focused_verification=True),
  )
  runner = Runner(checkpoint="memory", memory="disabled")
  session = await runner._create_session(agent, deps=None, session_id="verified-external")
  await session.memory.add("fix it", MemoryRole.USER)
  await session.memory.add(
    "",
    MemoryRole.ASSISTANT,
    tool_calls=[{
      "id": "verify_call",
      "function": {
        "name": "bash",
        "arguments": '{"command":"NONOKA_VERIFY=focused pytest -q tests/test_api.py"}',
      },
    }],
  )
  session.status = "paused"
  await runner.checkpoint_store.save_session(session.session_id, session.to_state())

  class _FinalLLM:
    async def chat_stream(self, messages, tools=None, **_kwargs):
      yield type("Chunk", (), {
        "content_delta": "verified",
        "tool_call_deltas": None,
        "finish_reason": "stop",
      })()

  runner._llm_cache["gpt-4o"] = _FinalLLM()
  events = []
  async for event in runner.resume_external_tools(
    agent,
    deps=None,
    session_id=session.session_id,
    results={"verify_call": ExternalToolReceipt(
      result="2 passed in 0.1s",
      exit_code=0,
      completeness=ObservationCompleteness.COMPLETE,
      verification=VerificationReceipt(
        status=VerificationStatus.PASSED,
        level=VerificationLevel.FOCUSED,
        kind=VerificationKind.TEST,
        command="NONOKA_VERIFY=focused pytest -q tests/test_api.py",
        cwd="/workspace",
        exit_code=0,
        completeness=ObservationCompleteness.COMPLETE,
        collected_tests=2,
      ),
    )},
  ):
    events.append(event)

  assert next(event for event in events if event.type == "final").data["success"] is True
  state = await runner.checkpoint_store.load_session(session.session_id)
  assert state.runtime_state.usage.focused_verification_status == "passed"


@pytest.mark.asyncio
async def test_mixed_local_and_external_calls_execute_local_before_host_pause():
  """A local result survives an external pause and is not sent to the host."""
  local = _LocalCapability()
  local.host_visible = False
  external = _ExternalCapability("bash", "Run commands", {"type": "object", "properties": {}})
  agent = Agent(model="gpt-4o", tools=[local, external], system_prompt="test")
  runner = Runner(checkpoint="memory", memory="disabled")

  class _FakeLLM:
    async def chat_stream(self, messages, tools=None, **_kwargs):
      yield type("Chunk", (), {
        "content_delta": "",
        "tool_call_deltas": [
          {"index": 0, "id": "local_1", "function": {"name": "local_echo", "arguments": '{"value":"evidence"}'}},
          {"index": 1, "id": "external_1", "function": {"name": "bash", "arguments": '{"command":"pwd"}'}},
        ],
        "finish_reason": "tool_calls",
      })()

  runner._llm_cache["gpt-4o"] = _FakeLLM()
  runner.llm = _FakeLLM()
  events: list[StreamEvent] = []
  async for event in runner.run_react_stream(agent, "inspect", deps=None, session_id="mixed-tools"):
    events.append(event)

  starts = [event for event in events if event.type == "tool_call_start"]
  assert len(starts) == 1
  assert [call["function"]["name"] for call in starts[0].data["tool_calls"]] == ["bash"]
  local_results = [event for event in events if event.type == "tool_call_result"]
  assert len(local_results) == 1
  assert local_results[0].data["name"] == "local_echo"
  assert local_results[0].data["result"] == {"echo": "evidence"}
  assert local_results[0].data["host_visible"] is False
  final = next(event for event in events if event.type == "final")
  assert final.data["requires_external_execution"] is True

  state = await runner.checkpoint_store.load_session("mixed-tools")
  assert state is not None
  tool_entries = [entry for entry in state.memory_entries if entry["role"] == MemoryRole.TOOL]
  assert len(tool_entries) == 1
  assert "evidence" in tool_entries[0]["content"]


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


def test_external_workspace_mutation_requires_host_attestation():
  capability = ExternalCapability(
    name="write_file", description="Write a file", parameters={"type": "object"},
    execution=ToolExecution(mutates_workspace=True),
  )

  assert capability.requires_workspace_attestation is True
  receipt = ExternalToolReceipt(
    result="written", workspace=WorkspaceAttestation(
      root="/workspace", before_digest="before", after_digest="after", created=("answer.txt",),
    ),
  )
  assert receipt.workspace.created == ("answer.txt",)


def test_workspace_attestation_round_trips_policy_enforcement():
  receipt = ExternalToolReceipt.from_value({
    "result": "blocked",
    "host": "test",
    "workspace": {
      "root": "/workspace",
      "before_digest": "same",
      "after_digest": "same",
      "policy_violations": ["fixture.db"],
      "restored_paths": ["fixture.db"],
    },
  })

  assert receipt.workspace is not None
  assert receipt.workspace.policy_violations == ("fixture.db",)
  assert receipt.workspace.restored_paths == ("fixture.db",)


def test_external_receipt_round_trips_observation_completeness():
  receipt = ExternalToolReceipt.from_value({
    "result": "bounded result",
    "host": "test",
    "completeness": "complete",
  })

  assert receipt.completeness == ObservationCompleteness.COMPLETE


def test_external_receipt_round_trips_verification():
  receipt = ExternalToolReceipt.from_value({
    "result": "2 passed",
    "exit_code": 0,
    "completeness": "complete",
    "verification": {
      "status": "passed",
      "level": "focused",
      "kind": "test",
      "command": "pytest -q tests/test_api.py",
      "cwd": "/workspace",
      "exit_code": 0,
      "completeness": "complete",
      "collected_tests": 2,
    },
  })

  assert receipt.verification == VerificationReceipt(
    status=VerificationStatus.PASSED,
    level=VerificationLevel.FOCUSED,
    kind=VerificationKind.TEST,
    command="pytest -q tests/test_api.py",
    cwd="/workspace",
    exit_code=0,
    completeness=ObservationCompleteness.COMPLETE,
    collected_tests=2,
  )


def test_legacy_truncated_receipt_maps_to_partial_observation():
  receipt = ExternalToolReceipt.from_value({
    "result": "preview",
    "host": "legacy",
    "truncated": True,
  })

  assert receipt.completeness == ObservationCompleteness.PARTIAL


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
    async def chat_stream(self, messages, tools=None, **_kwargs):
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
  final = next(event for event in events if event.type == "final")
  assert "runtime" in final.data
  assert "trace" not in final.data
  assert "final" in types

  final = next(e for e in events if e.type == "final")
  assert final.data.get("requires_external_execution") is True


@pytest.mark.asyncio
async def test_external_progress_extension_runs_after_host_receipts():
  """Guidance is persisted after TOOL receipts without breaking adjacency."""
  agent = Agent(
    model="gpt-4o",
    tools=[_ExternalCapability("grep", "Search files", {"type": "object"})],
    extensions=[WorkspaceProgressExtension(max_exploration_turns=1)],
  )
  runner = Runner(checkpoint="memory", memory="disabled")

  class _FakeLLM:
    async def chat_stream(self, messages, tools=None, **_kwargs):
      yield type("Chunk", (), {
        "content_delta": "",
        "tool_call_deltas": [{
          "index": 0,
          "id": "search_1",
          "function": {"name": "grep", "arguments": '{"pattern":"needle"}'},
        }],
        "finish_reason": "tool_calls",
      })()

  runner._llm_cache["gpt-4o"] = _FakeLLM()
  events = []
  async for event in runner.run_react_stream(
    agent, "change the workspace", deps=None, session_id="external-progress",
  ):
    events.append(event)

  assert next(event for event in events if event.type == "final").data[
    "requires_external_execution"
  ] is True
  state = await runner.checkpoint_store.load_session("external-progress")
  assert not any(
    "Stop broad exploration" in entry["content"]
    for entry in state.memory_entries
  )

  seen_messages = []

  class _FinalLLM:
    async def chat_stream(self, messages, tools=None, **_kwargs):
      seen_messages.extend(messages)
      yield type("Chunk", (), {
        "content_delta": "done",
        "tool_call_deltas": None,
        "finish_reason": "stop",
      })()

  runner._llm_cache["gpt-4o"] = _FinalLLM()
  async for _ in runner.resume_external_tools(
    agent,
    deps=None,
    session_id="external-progress",
    results={"search_1": ExternalToolReceipt(result="no matches")},
  ):
    pass

  state = await runner.checkpoint_store.load_session("external-progress")
  assistant_index = next(
    index for index, entry in enumerate(state.memory_entries)
    if entry["role"] == MemoryRole.ASSISTANT and entry["metadata"].get("tool_calls")
  )
  assert state.memory_entries[assistant_index + 1]["role"] == MemoryRole.TOOL
  feedback = [entry["content"] for entry in state.memory_entries]
  assert any("Stop broad exploration" in message for message in feedback)
  assert any(
    message.role == "system" and "Stop broad exploration" in message.content
    for message in seen_messages
  )


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
    async def chat_stream(self, messages, tools=None, **_kwargs):
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
  assert "trace" in final.data
  assert final.data.get("success") is True


@pytest.mark.asyncio
async def test_external_resume_does_not_reset_session_tool_budget():
  agent = Agent(
    model="gpt-4o",
    tools=[_ExternalCapability("bash", "Run commands", {"type": "object"})],
    runtime_limits=RuntimeLimits(max_model_turns=4, max_tool_calls=1),
  )
  runner = Runner(checkpoint="memory", memory="disabled")
  session = await runner._create_session(agent, deps=None, session_id="budget-sess")
  await session.memory.add("run commands", MemoryRole.USER)
  await session.memory.add(
    "", MemoryRole.ASSISTANT,
    tool_calls=[{"id": "call_1", "function": {"name": "bash", "arguments": "{}"}}],
  )
  session.runtime_state.usage.model_turns = 1
  session.runtime_state.usage.tool_calls = 1
  session.status = "paused"
  await runner.checkpoint_store.save_session(session.session_id, session.to_state())

  class _FakeLLM:
    async def chat_stream(self, messages, tools=None, **_kwargs):
      yield type("Chunk", (), {
        "content_delta": "", "finish_reason": "tool_calls",
        "tool_call_deltas": [{
          "index": 0, "id": "call_2",
          "function": {"name": "bash", "arguments": '{"command":"pwd"}'},
        }],
      })()

  runner._llm_cache["gpt-4o"] = _FakeLLM()
  events = []
  async for event in runner.resume_external_tools(
    agent, deps=None, session_id=session.session_id, results={"call_1": "ok"},
  ):
    events.append(event)

  error = next(event for event in events if event.type == "error")
  assert error.data["termination"]["reason"] == "tool_budget_exhausted"
  state = await runner.checkpoint_store.load_session(session.session_id)
  assert state.runtime_state.usage.tool_calls == 1


@pytest.mark.asyncio
async def test_external_result_is_bounded_before_memory_insertion():
  agent = Agent(
    model="gpt-4o",
    tools=[_ExternalCapability("bash", "Run commands", {"type": "object"})],
    runtime_limits=RuntimeLimits(max_external_result_bytes=1024, max_context_bytes=8192),
  )
  runner = Runner(checkpoint="memory", memory="disabled")
  session = await runner._create_session(agent, deps=None, session_id="large-result")
  await session.memory.add("run", MemoryRole.USER)
  await session.memory.add(
    "", MemoryRole.ASSISTANT,
    tool_calls=[{"id": "call_1", "function": {"name": "bash", "arguments": "{}"}}],
  )
  session.status = "paused"
  await runner.checkpoint_store.save_session(session.session_id, session.to_state())

  class _FakeLLM:
    async def chat_stream(self, messages, tools=None, **_kwargs):
      yield type("Chunk", (), {
        "content_delta": "done", "tool_call_deltas": None, "finish_reason": "stop",
      })()

  runner._llm_cache["gpt-4o"] = _FakeLLM()
  async for _ in runner.resume_external_tools(
    agent, deps=None, session_id=session.session_id,
    results={"call_1": ExternalToolReceipt(
      result="x" * 10000, artifact_ref="trace://large.txt", original_bytes=10000,
    )},
  ):
    pass

  state = await runner.checkpoint_store.load_session(session.session_id)
  tool_entry = next(entry for entry in state.memory_entries if entry["role"] == "tool")
  assert len(tool_entry["content"].encode()) <= 1100
  assert tool_entry["metadata"]["artifact_ref"] == "trace://large.txt"
  assert tool_entry["metadata"]["truncated"] is True


@pytest.mark.asyncio
async def test_external_effect_attestation_updates_checkpointed_runtime_evidence():
  agent = Agent(
    model="gpt-4o",
    tools=[_ExternalCapability("bash", "Run commands", {"type": "object"})],
    completion_contract=CompletionContract(require_observed_effect=True),
  )
  runner = Runner(checkpoint="memory", memory="disabled")
  session = await runner._create_session(agent, deps=None, session_id="effect-evidence")
  await session.memory.add("configure the service", MemoryRole.USER)
  await session.memory.add(
    "", MemoryRole.ASSISTANT,
    tool_calls=[{
      "id": "call_1",
      "function": {"name": "bash", "arguments": '{"command":"configure-service"}'},
    }],
  )
  session.status = "paused"
  await runner.checkpoint_store.save_session(session.session_id, session.to_state())

  class _FakeLLM:
    async def chat_stream(self, messages, tools=None, **_kwargs):
      yield type("Chunk", (), {
        "content_delta": "configured", "tool_call_deltas": None, "finish_reason": "stop",
      })()

  runner._llm_cache["gpt-4o"] = _FakeLLM()
  async for _ in runner.resume_external_tools(
    agent, deps=None, session_id=session.session_id,
    results={"call_1": ExternalToolReceipt(
      result="ok",
      exit_code=0,
      completeness=ObservationCompleteness.COMPLETE,
      effect=EffectAttestation(changed=True, scope="system", collector="test-host"),
    )},
  ):
    pass

  state = await runner.checkpoint_store.load_session(session.session_id)
  usage = state.runtime_state.usage
  assert usage.effect_count == 1
  assert usage.last_effect_at_observation == 1
  assert usage.successful_command_count == 1
  assert usage.last_successful_command == "configure-service"
  assert usage.last_successful_command_at_observation == 1


@pytest.mark.asyncio
async def test_partial_observation_requires_fresh_complete_follow_up():
  agent = Agent(
    model="gpt-4o",
    tools=[_ExternalCapability("inspect", "Inspect data", {"type": "object"})],
    completion_contract=CompletionContract(
      require_complete_observations=True,
      max_corrections=1,
    ),
  )
  runner = Runner(checkpoint="memory", memory="disabled")
  session = await runner._create_session(agent, deps=None, session_id="observation-contract")
  await session.memory.add("inspect everything", MemoryRole.USER)
  await session.memory.add(
    "", MemoryRole.ASSISTANT,
    tool_calls=[{"id": "call_1", "function": {"name": "inspect", "arguments": "{}"}}],
  )
  session.status = "paused"
  await runner.checkpoint_store.save_session(session.session_id, session.to_state())

  class _ReviewLLM:
    calls = 0

    async def chat_stream(self, messages, tools=None, **_kwargs):
      self.calls += 1
      if self.calls == 1:
        yield type("Chunk", (), {
          "content_delta": "looks done", "tool_call_deltas": None, "finish_reason": "stop",
        })()
      else:
        yield type("Chunk", (), {
          "content_delta": "", "finish_reason": "tool_calls",
          "tool_call_deltas": [{
            "index": 0, "id": "call_2",
            "function": {"name": "inspect", "arguments": '{"scope":"bounded"}'},
          }],
        })()

  runner._llm_cache["gpt-4o"] = _ReviewLLM()
  events = []
  async for event in runner.resume_external_tools(
    agent,
    deps=None,
    session_id=session.session_id,
    results={"call_1": ExternalToolReceipt(
      result="preview",
      artifact_ref="trace://preview.txt",
      completeness=ObservationCompleteness.PARTIAL,
    )},
  ):
    events.append(event)

  assert any(event.type == "tool_call_start" for event in events)
  state = await runner.checkpoint_store.load_session(session.session_id)
  first_tool = next(entry for entry in state.memory_entries if entry["role"] == "tool")
  assert first_tool["content"].startswith("[Partial observation]")
  assert first_tool["metadata"]["completeness"] == "partial"
  assert state.runtime_state.usage.partial_observation_count == 1
  assert state.runtime_state.usage.observation_feedback_after == 1

  class _FinalLLM:
    async def chat_stream(self, messages, tools=None, **_kwargs):
      yield type("Chunk", (), {
        "content_delta": "verified", "tool_call_deltas": None, "finish_reason": "stop",
      })()

  runner._llm_cache["gpt-4o"] = _FinalLLM()
  resumed_events = []
  async for event in runner.resume_external_tools(
    agent,
    deps=None,
    session_id=session.session_id,
    results={"call_2": ExternalToolReceipt(
      result="full bounded result",
      completeness=ObservationCompleteness.COMPLETE,
    )},
  ):
    resumed_events.append(event)

  final = next(event for event in resumed_events if event.type == "final")
  assert final.data["success"] is True
  final_state = await runner.checkpoint_store.load_session(session.session_id)
  assert final_state.runtime_state.usage.last_complete_observation_at == 2
  assert final_state.runtime_state.usage.complete_observations_after_partial == 1


@pytest.mark.asyncio
async def test_restored_workspace_policy_violation_becomes_agent_feedback():
  capability = ExternalCapability(
    name="host_tool", description="Host tool", parameters={"type": "object"},
    audit_required=True,
  )
  agent = Agent(model="gpt-4o", tools=[capability])
  runner = Runner(checkpoint="memory", memory="disabled")
  session = await runner._create_session(agent, deps=None, session_id="policy-sess")
  await session.memory.add("use protected input safely", MemoryRole.USER)
  await session.memory.add(
    "", MemoryRole.ASSISTANT,
    tool_calls=[{"id": "call_1", "function": {"name": "host_tool", "arguments": "{}"}}],
  )
  session.status = "paused"
  await runner.checkpoint_store.save_session(session.session_id, session.to_state())

  class _FakeLLM:
    async def chat_stream(self, messages, tools=None, **_kwargs):
      yield type("Chunk", (), {
        "content_delta": "done", "tool_call_deltas": None, "finish_reason": "stop",
      })()

  runner._llm_cache["gpt-4o"] = _FakeLLM()
  async for _ in runner.resume_external_tools(
    agent,
    deps=None,
    session_id=session.session_id,
    results={"call_1": ExternalToolReceipt(
      result="command completed",
      workspace=WorkspaceAttestation(
        root="/workspace",
        before_digest="same",
        after_digest="same",
        policy_violations=("fixture.db",),
        restored_paths=("fixture.db",),
      ),
    )},
  ):
    pass

  state = await runner.checkpoint_store.load_session(session.session_id)
  tool_entry = next(entry for entry in state.memory_entries if entry["role"] == "tool")
  assert tool_entry["content"].startswith("[Execution policy]")
  assert "fixture.db" in tool_entry["content"]
  assert state.runtime_state.usage.policy_violation_count == 1
  assert state.runtime_state.usage.policy_violations == ["fixture.db"]


@pytest.mark.asyncio
async def test_resume_external_mutation_rejects_missing_workspace_attestation():
  capability = ExternalCapability(
    name="write_file", description="Write", parameters={"type": "object"},
    execution=ToolExecution(mutates_workspace=True),
  )
  agent = Agent(model="gpt-4o", tools=[capability])
  runner = Runner(checkpoint="memory", memory="disabled")
  session = await runner._create_session(agent, deps=None, session_id="audit-sess")
  await session.memory.add("write", MemoryRole.USER)
  await session.memory.add(
    "", MemoryRole.ASSISTANT,
    tool_calls=[{"id": "call_1", "function": {"name": "write_file", "arguments": "{}"}}],
  )
  session.status = "paused"
  await runner.checkpoint_store.save_session("audit-sess", session.to_state())

  with pytest.raises(ValueError, match="workspace attestation"):
    async for _ in runner.resume_external_tools(
      agent, deps=None, session_id="audit-sess", results={"call_1": "written"},
    ):
      pass
