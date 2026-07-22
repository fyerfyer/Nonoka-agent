from __future__ import annotations

import asyncio

import pytest

from nonoka.core.execution import ToolExecution, ToolExecutionCoordinator
from nonoka.core.trace import ExecutionTrace
from nonoka import Agent, Runner
from nonoka.core.llm import LLMResponse


class _Capability:
  def __init__(self, execution: ToolExecution | None = None) -> None:
    if execution is not None:
      self.execution = execution


@pytest.mark.asyncio
async def test_execution_coordinator_parallelizes_only_explicit_reads():
  read = _Capability(ToolExecution(read_only=True))
  write = _Capability(ToolExecution(mutates_workspace=True, stateful_action=True))
  calls = [{"name": "read-a"}, {"name": "read-b"}, {"name": "write"}, {"name": "read-c"}]
  capabilities = {"read-a": read, "read-b": read, "write": write, "read-c": read}
  events: list[str] = []
  active = 0
  max_active = 0

  async def invoke(call):
    nonlocal active, max_active
    active += 1
    max_active = max(max_active, active)
    events.append(f"start:{call['name']}")
    await asyncio.sleep(0.01)
    events.append(f"end:{call['name']}")
    active -= 1
    return call["name"]

  results = await ToolExecutionCoordinator(2).execute(
    calls, lambda call: capabilities[call["name"]], invoke,
  )

  assert results == ["read-a", "read-b", "write", "read-c"]
  assert max_active == 2
  assert events.index("start:write") > events.index("end:read-a")
  assert events.index("start:write") > events.index("end:read-b")


@pytest.mark.asyncio
async def test_execution_coordinator_serializes_unknown_capabilities():
  calls = [{"name": "first"}, {"name": "second"}]
  active = 0
  max_active = 0

  async def invoke(_call):
    nonlocal active, max_active
    active += 1
    max_active = max(max_active, active)
    await asyncio.sleep(0.01)
    active -= 1
    return "ok"

  await ToolExecutionCoordinator(10).execute(calls, lambda _call: _Capability(), invoke)
  assert max_active == 1


def test_execution_trace_redacts_credentials_and_bounds_output():
  trace = ExecutionTrace()
  trace.record_generation(api_key="secret", prompt="Bearer visible-token", max_tokens=256)
  trace.record_tool_start("tc", "tool", {"password": "pw"}, ToolExecution(read_only=True))
  payload = trace.to_dict()

  assert payload["generation"]["api_key"] == "[REDACTED]"
  assert payload["generation"]["prompt"] == "[REDACTED]"
  assert payload["generation"]["max_tokens"] == 256
  assert payload["tool_calls"][0]["arguments"]["password"] == "[REDACTED]"


@pytest.mark.asyncio
async def test_runner_attaches_serializable_trace_to_result():
  class Provider:
    async def chat(self, **_kwargs):
      return LLMResponse(content="done", usage={"prompt_tokens": 3, "completion_tokens": 2})

  runner = Runner(checkpoint="memory", memory="in_memory")
  runner._create_llm = lambda _agent: Provider()  # type: ignore[method-assign]
  result = await runner.run_react(Agent(model="fake"), "hello", deps=None)

  assert result.success is True
  assert result.trace is not None
  assert result.trace["generation"]["model"] == "fake"
  assert result.trace["turns"][0]["response"]["usage"]["prompt_tokens"] == 3
  assert result.trace["termination"]["success"] is True
