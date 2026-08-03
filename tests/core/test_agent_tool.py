from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from nonoka.core.agent import Agent
from nonoka.core.agent_tool import AgentTool, MemoryStrategy
from nonoka.core.context import RunContext
from nonoka.core.hooks import Hooks
from nonoka.core.memory import MemoryRole
from nonoka.core.runner import Runner
from nonoka.core.session import Session, SessionStatus
from nonoka.core.tool import tool
from nonoka.core.types import RunResult


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _make_mock_runner(
  response_content: str = "mocked response",
  tool_calls: list[dict[str, Any]] | None = None,
) -> Runner:
  """Create a Runner with mocked LLM so no real network calls happen.

  The mock provider is returned for ANY agent model via _create_llm override.
  """
  runner = Runner(checkpoint="memory", memory="in_memory")
  provider = MagicMock()
  provider.chat = AsyncMock(
    return_value=MagicMock(
      content=response_content,
      tool_calls=tool_calls,
      usage={},
    )
  )
  provider.chat_stream = AsyncMock(return_value=iter([]))
  # Override _create_llm so ANY agent model gets the mock
  runner._create_llm = lambda agent: provider  # type: ignore[method-assign]
  runner.llm = provider
  return runner


def _setup_mock_runner(
  runner: Runner,
  response_content: str = "mocked response",
  tool_calls: list[dict[str, Any]] | None = None,
) -> MagicMock:
  """Attach a mock LLM provider to an existing Runner."""
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
  return provider


# --------------------------------------------------------------------------- #
# 1. Construction and schema
# --------------------------------------------------------------------------- #

def test_agent_tool_default_name_and_description():
  """Defaults should be sensible when name/description are omitted."""
  agent = Agent(model="gpt-4o", tools=[])
  at = AgentTool(agent=agent)

  assert at.name == "agent_gpt-4o"
  assert "gpt-4o" in at.description
  assert at.memory_strategy == MemoryStrategy.ISOLATE
  assert at.max_depth == 3


def test_agent_tool_custom_name_and_description():
  """Custom name/description should be honoured."""
  agent = Agent(model="gpt-4o", tools=[])
  at = AgentTool(
    agent=agent,
    name="security_reviewer",
    description="Reviews code for security issues.",
    memory_strategy=MemoryStrategy.INHERIT,
    max_depth=5,
    inherit_memory_count=10,
  )

  assert at.name == "security_reviewer"
  assert at.description == "Reviews code for security issues."
  assert at.memory_strategy == MemoryStrategy.INHERIT
  assert at.max_depth == 5
  assert at.inherit_memory_count == 10


def test_agent_tool_parameters_schema():
  """The JSON schema exposed to the LLM must have 'task' and optional 'context'."""
  at = AgentTool(agent=Agent(model="test", tools=[]))
  schema = at.parameters

  assert schema["type"] == "object"
  assert "task" in schema["properties"]
  assert "context" in schema["properties"]
  assert schema["required"] == ["task"]
  assert schema["properties"]["task"]["type"] == "string"
  assert schema["properties"]["context"]["type"] == "string"


def test_agent_tool_to_json_schema():
  """to_json_schema must be OpenAI-compatible function schema."""
  at = AgentTool(agent=Agent(model="test", tools=[]), name="sub", description="desc")
  schema = at.to_json_schema()

  assert schema["type"] == "function"
  assert schema["function"]["name"] == "sub"
  assert schema["function"]["description"] == "desc"
  assert "parameters" in schema["function"]


def test_memory_strategy_from_string():
  """MemoryStrategy should be constructible from a plain string."""
  at = AgentTool(agent=Agent(model="test", tools=[]), memory_strategy="inherit")
  assert at.memory_strategy == MemoryStrategy.INHERIT

  at2 = AgentTool(agent=Agent(model="test", tools=[]), memory_strategy="share")
  assert at2.memory_strategy == MemoryStrategy.SHARE


# --------------------------------------------------------------------------- #
# 2. Depth limiting
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_agent_tool_depth_limit_blocks_execution():
  """When session._agent_depth >= max_depth, invoke should return an error."""
  at = AgentTool(agent=Agent(model="test", tools=[]), max_depth=2)

  # Create a parent session at depth 2 (already at limit)
  runner = _make_mock_runner()
  session = await runner._create_session(Agent(model="test", tools=[]), deps=None)
  object.__setattr__(session, "_agent_depth", 2)
  ctx = RunContext(session)

  result = await at.invoke(ctx, {"task": "do something"})

  assert isinstance(result, dict)
  assert "error" in result
  assert "depth" in result["error"].lower()
  # The sub-agent should NOT have been executed
  runner.llm.chat.assert_not_called()


@pytest.mark.asyncio
async def test_agent_tool_executes_below_depth_limit():
  """When session._agent_depth < max_depth, sub-agent should run."""
  at = AgentTool(agent=Agent(model="test", tools=[]), max_depth=2)

  runner = _make_mock_runner()
  session = await runner._create_session(Agent(model="test", tools=[]), deps=None)
  object.__setattr__(session, "_agent_depth", 0)
  ctx = RunContext(session)

  await at.invoke(ctx, {"task": "do something"})

  # Should have called the LLM (sub-agent ran)
  runner.llm.chat.assert_called()


@pytest.mark.asyncio
async def test_agent_tool_depth_0_by_default():
  """Sessions without _agent_depth should be treated as depth 0."""
  at = AgentTool(agent=Agent(model="test", tools=[]), max_depth=1)

  runner = _make_mock_runner()
  session = await runner._create_session(Agent(model="test", tools=[]), deps=None)
  # Do NOT set _agent_depth — it should default to 0
  assert not hasattr(session, "_agent_depth")
  ctx = RunContext(session)

  await at.invoke(ctx, {"task": "do something"})

  # Should succeed because depth 0 < max_depth 1
  runner.llm.chat.assert_called()


# --------------------------------------------------------------------------- #
# 3. Cancel propagation
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_agent_tool_respects_parent_cancellation():
  """If the parent session is cancelled, the sub-agent should not run."""
  at = AgentTool(agent=Agent(model="test", tools=[]))

  runner = _make_mock_runner()
  session = await runner._create_session(Agent(model="test", tools=[]), deps=None)
  session.cancel()
  ctx = RunContext(session)

  result = await at.invoke(ctx, {"task": "do something"})

  assert isinstance(result, dict)
  assert "error" in result
  assert "cancel" in result["error"].lower()
  runner.llm.chat.assert_not_called()


@pytest.mark.asyncio
async def test_agent_tool_propagates_cancellation_while_child_is_running():
  """Cancelling the parent should stop an already-running child session."""
  started = asyncio.Event()

  async def slow_chat(*args, **kwargs):
    started.set()
    await asyncio.Event().wait()

  provider = MagicMock()
  provider.chat = AsyncMock(side_effect=slow_chat)
  runner = Runner(checkpoint="memory", memory="in_memory")
  runner._create_llm = lambda agent: provider  # type: ignore[method-assign]
  runner.llm = provider
  parent = await runner._create_session(Agent(model="parent", tools=[]), deps=None)
  invocation = asyncio.create_task(
    AgentTool(agent=Agent(model="child", tools=[])).invoke(
      RunContext(parent), {"task": "wait"}
    )
  )

  await started.wait()
  parent.cancel()
  result = await invocation

  assert result["success"] is False
  assert result["error_type"] == "cancelled"


# --------------------------------------------------------------------------- #
# 4. Result extraction
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_agent_tool_default_result_extractor_returns_data():
  """By default the tool should return the sub-agent's result.data."""
  at = AgentTool(agent=Agent(model="test", tools=[]))

  runner = _make_mock_runner()
  # Make LLM return a simple answer
  runner.llm.chat.return_value = MagicMock(
    content='{"answer": 42}', tool_calls=None, usage={}
  )

  session = await runner._create_session(Agent(model="test", tools=[]), deps=None)
  ctx = RunContext(session)

  result = await at.invoke(ctx, {"task": "what is the answer"})

  # With mocked LLM returning content, the sub-agent succeeds
  # and default extractor returns result.data (the content string)
  assert result is not None


@pytest.mark.asyncio
async def test_agent_tool_custom_result_extractor():
  """Users can provide a custom extractor to shape the output."""
  def extract(result: RunResult) -> dict:
    return {
      "ok": result.success,
      "payload": result.data,
      "turns": result.session.turn_count if result.session else 0,
    }

  at = AgentTool(
    agent=Agent(model="test", tools=[]),
    result_extractor=extract,
  )

  runner = _make_mock_runner()
  runner.llm.chat.return_value = MagicMock(
    content="hello", tool_calls=None, usage={}
  )

  session = await runner._create_session(Agent(model="test", tools=[]), deps=None)
  ctx = RunContext(session)

  result = await at.invoke(ctx, {"task": "greet"})

  assert isinstance(result, dict)
  assert "ok" in result
  assert "payload" in result
  assert "turns" in result


@pytest.mark.asyncio
async def test_agent_tool_selects_child_model_and_restores_parent_provider():
  parent_provider = MagicMock()
  child_provider = MagicMock()
  child_provider.chat = AsyncMock(
    return_value=MagicMock(content="child answer", tool_calls=None, usage={})
  )
  providers = {"parent-model": parent_provider, "child-model": child_provider}
  runner = Runner(checkpoint="memory", memory="in_memory")
  runner._create_llm = lambda agent: providers[agent.model]  # type: ignore[method-assign]
  runner.llm = parent_provider
  parent = await runner._create_session(Agent(model="parent-model", tools=[]), deps=None)

  result = await AgentTool(agent=Agent(model="child-model", tools=[])).invoke(
    RunContext(parent), {"task": "answer"}
  )

  assert result == "child answer"
  child_provider.chat.assert_awaited()
  assert runner.llm is parent_provider


@pytest.mark.asyncio
async def test_concurrent_agent_tools_keep_providers_task_local():
  both_started = asyncio.Event()
  started = 0

  def provider(label):
    mock = MagicMock()

    async def chat(**_kwargs):
      nonlocal started
      started += 1
      if started == 2:
        both_started.set()
      await asyncio.wait_for(both_started.wait(), timeout=1)
      return MagicMock(content=label, tool_calls=None, usage={})

    mock.chat = AsyncMock(side_effect=chat)
    return mock

  providers = {"child-a": provider("A"), "child-b": provider("B")}
  runner = Runner(checkpoint="memory", memory="in_memory")
  runner._create_llm = lambda agent: providers[agent.model]  # type: ignore[method-assign]
  parent_a = await runner._create_session(Agent(model="parent", tools=[]), deps=None)
  parent_b = await runner._create_session(Agent(model="parent", tools=[]), deps=None)

  first, second = await asyncio.gather(
    AgentTool(agent=Agent(model="child-a", tools=[])).invoke(
      RunContext(parent_a), {"task": "first"}
    ),
    AgentTool(agent=Agent(model="child-b", tools=[])).invoke(
      RunContext(parent_b), {"task": "second"}
    ),
  )

  assert (first, second) == ("A", "B")


@pytest.mark.asyncio
async def test_agent_tool_emits_child_session_hooks_with_parent_link():
  starts = []
  finishes = []

  async def on_start(ctx):
    starts.append(ctx)

  async def on_end(ctx, result):
    finishes.append((ctx, result))

  runner = _make_mock_runner(response_content="done")
  runner.hooks = Hooks(on_session_start=on_start, on_session_end=on_end)
  parent = await runner._create_session(Agent(model="parent", tools=[]), deps=None)

  await AgentTool(agent=Agent(model="child", tools=[])).invoke(
    RunContext(parent), {"task": "review"}
  )

  assert len(starts) == 1
  assert len(finishes) == 1
  child = starts[0].session
  assert child.session_id != parent.session_id
  assert child._parent_session_id == parent.session_id
  assert finishes[0][1].success is True


@pytest.mark.asyncio
async def test_agent_tool_extractor_on_failure():
  """When sub-agent fails, default extractor should include error metadata."""
  at = AgentTool(agent=Agent(model="test", tools=[]))

  runner = _make_mock_runner()
  # Simulate LLM failure by raising an exception
  runner.llm.chat.side_effect = RuntimeError("LLM exploded")

  session = await runner._create_session(Agent(model="test", tools=[]), deps=None)
  ctx = RunContext(session)

  result = await at.invoke(ctx, {"task": "fail me"})

  assert isinstance(result, dict)
  assert result.get("success") is False
  assert "error" in result
  assert "error_type" in result


# --------------------------------------------------------------------------- #
# 5. Prompt construction
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_agent_tool_builds_prompt_with_task_only():
  """When only 'task' is provided, it should be the prompt verbatim."""
  at = AgentTool(agent=Agent(model="test", tools=[]))

  runner = _make_mock_runner()
  session = await runner._create_session(Agent(model="test", tools=[]), deps=None)
  ctx = RunContext(session)

  await at.invoke(ctx, {"task": "Calculate 2+2"})

  # The LLM should have been called with messages containing the task
  calls = runner.llm.chat.call_args_list
  assert len(calls) >= 1
  messages = calls[0].kwargs.get("messages") or calls[0][1].get("messages")
  # Messages should contain the task text somewhere
  all_content = " ".join(str(m.content) for m in messages)
  assert "Calculate 2+2" in all_content


@pytest.mark.asyncio
async def test_agent_tool_builds_prompt_with_task_and_context():
  """When 'context' is provided, it should be appended after the task."""
  at = AgentTool(agent=Agent(model="test", tools=[]))

  runner = _make_mock_runner()
  session = await runner._create_session(Agent(model="test", tools=[]), deps=None)
  ctx = RunContext(session)

  await at.invoke(ctx, {"task": "Review this code", "context": "Language: Python\nFile: main.py"})

  calls = runner.llm.chat.call_args_list
  messages = calls[0].kwargs.get("messages") or calls[0][1].get("messages")
  all_content = " ".join(str(m.content) for m in messages)
  assert "Review this code" in all_content
  assert "Language: Python" in all_content


# --------------------------------------------------------------------------- #
# 6. Memory strategies
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_agent_tool_isolate_strategy_no_parent_memory():
  """With ISOLATE, the child session should start with empty memory."""
  runner = Runner(checkpoint="memory", memory="in_memory")
  provider = _setup_mock_runner(runner, response_content="child answer")

  at = AgentTool(
    agent=Agent(model="test-model", tools=[]),
    memory_strategy=MemoryStrategy.ISOLATE,
  )

  # Parent session with some memory
  parent_agent = Agent(model="test-model", tools=[])
  parent_session = await runner._create_session(parent_agent, deps=None)
  await parent_session.memory.add("Parent secret", MemoryRole.USER)
  await parent_session.memory.add("Parent reply", MemoryRole.ASSISTANT)

  ctx = RunContext(parent_session)
  result = await at.invoke(ctx, {"task": "do something"})

  # The child should have run (result is the LLM response content)
  assert result == "child answer"
  # The child's session should NOT have parent memory entries
  # We can verify by checking that the LLM messages didn't contain parent secrets
  calls = provider.chat.call_args_list
  messages = calls[0].kwargs.get("messages") or calls[0][1].get("messages")
  all_content = " ".join(str(m.content) for m in messages)
  assert "Parent secret" not in all_content


@pytest.mark.asyncio
async def test_agent_tool_inherit_strategy_copies_memory():
  """With INHERIT, the child session should copy last N parent memory entries."""
  runner = Runner(checkpoint="memory", memory="in_memory")
  provider = _setup_mock_runner(runner, response_content="child answer")

  at = AgentTool(
    agent=Agent(model="test-model", tools=[]),
    memory_strategy=MemoryStrategy.INHERIT,
    inherit_memory_count=2,
  )

  parent_agent = Agent(model="test-model", tools=[])
  parent_session = await runner._create_session(parent_agent, deps=None)
  await parent_session.memory.add("Entry 1", MemoryRole.USER)
  await parent_session.memory.add("Entry 2", MemoryRole.ASSISTANT)
  await parent_session.memory.add("Entry 3", MemoryRole.USER)

  ctx = RunContext(parent_session)
  await at.invoke(ctx, {"task": "do something"})

  # Child should have inherited last 2 entries
  calls = provider.chat.call_args_list
  messages = calls[0].kwargs.get("messages") or calls[0][1].get("messages")
  all_content = " ".join(str(m.content) for m in messages)
  # Entry 2 and Entry 3 should be present
  assert "Entry 2" in all_content
  assert "Entry 3" in all_content
  # Entry 1 was outside the window
  assert "Entry 1" not in all_content


@pytest.mark.asyncio
async def test_agent_tool_share_strategy_shares_memory_object():
  """With SHARE, the child should use the exact same WorkingMemory instance."""
  runner = Runner(checkpoint="memory", memory="in_memory")
  provider = _setup_mock_runner(runner, response_content="child answer")

  at = AgentTool(
    agent=Agent(model="test-model", tools=[]),
    memory_strategy=MemoryStrategy.SHARE,
  )

  parent_agent = Agent(model="test-model", tools=[])
  parent_session = await runner._create_session(parent_agent, deps=None)
  await parent_session.memory.add("Shared context", MemoryRole.USER)

  ctx = RunContext(parent_session)
  await at.invoke(ctx, {"task": "do something"})

  # Child and parent share memory, so child sees parent entries
  calls = provider.chat.call_args_list
  messages = calls[0].kwargs.get("messages") or calls[0][1].get("messages")
  all_content = " ".join(str(m.content) for m in messages)
  assert "Shared context" in all_content


# --------------------------------------------------------------------------- #
# 7. Deps inheritance
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_agent_tool_inherits_parent_deps():
  """The sub-agent should receive the parent's deps object."""
  class FakeDeps:
    def __init__(self, value: str):
      self.value = value

  runner = Runner(checkpoint="memory", memory="in_memory")

  # Create a tool that reads deps to verify inheritance
  deps_value_captured = None

  @tool
  async def capture_deps(ctx: RunContext) -> str:
    nonlocal deps_value_captured
    deps_value_captured = ctx.deps.value if hasattr(ctx.deps, "value") else None
    return f"captured: {deps_value_captured}"

  sub_agent = Agent(model="test-model", tools=[capture_deps])
  at2 = AgentTool(agent=sub_agent)

  parent_agent = Agent(model="test-model", tools=[at2])
  deps = FakeDeps(value="inherited_value")

  # Mock LLM so the sub-agent calls capture_deps on turn 1,
  # then returns final answer on turn 2.
  call_count = 0
  def mock_chat(*args, **kwargs):
    nonlocal call_count
    call_count += 1
    if call_count == 1:
      return MagicMock(
        content="",
        tool_calls=[{
          "id": "call_1",
          "type": "function",
          "function": {"name": "capture_deps", "arguments": "{}"},
        }],
        usage={},
      )
    return MagicMock(content="done", tool_calls=None, usage={})

  provider = MagicMock()
  provider.chat = AsyncMock(side_effect=mock_chat)
  runner._create_llm = lambda agent: provider  # type: ignore[method-assign]
  runner.llm = provider

  parent_session = await runner._create_session(parent_agent, deps=deps)
  ctx = RunContext(parent_session)

  # Directly invoke the sub-agent tool with the parent context
  await at2.invoke(ctx, {"task": "capture deps"})

  # The sub-agent should have run with the same deps
  assert deps_value_captured == "inherited_value"


# --------------------------------------------------------------------------- #
# 8. Runner resolution
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_agent_tool_resolves_runner_from_session():
  """When session has _runner_ref, AgentTool should use that runner."""
  runner = _make_mock_runner()
  session = await runner._create_session(Agent(model="test", tools=[]), deps=None)

  # Verify _runner_ref is set by Runner._create_session
  assert hasattr(session, "_runner_ref")
  assert session._runner_ref() is runner


@pytest.mark.asyncio
async def test_agent_tool_fallback_runner_when_no_ref(monkeypatch):
  """When session lacks _runner_ref, AgentTool should create a default Runner."""
  at = AgentTool(agent=Agent(model="test", tools=[]))

  provider = MagicMock()
  provider.chat = AsyncMock(
    return_value=MagicMock(content="fallback answer", tool_calls=None, usage={})
  )
  monkeypatch.setattr(
    Runner,
    "_create_llm",
    lambda self, agent: provider,
  )

  # Manually create a session without _runner_ref
  session = Session(session_id="test-session", agent=Agent(model="test", tools=[]), deps=None)
  assert not hasattr(session, "_runner_ref")
  ctx = RunContext(session)

  # Should still work through the fallback Runner without a real model call.
  result = await at.invoke(ctx, {"task": "do something"})
  assert result == "fallback answer"


# --------------------------------------------------------------------------- #
# 9. Integration with Capability Protocol
# --------------------------------------------------------------------------- #

def test_agent_tool_satisfies_capability_protocol():
  """AgentTool must implement the Capability Protocol."""
  from nonoka.core.types import Capability

  at = AgentTool(agent=Agent(model="test", tools=[]))
  assert isinstance(at, Capability)


@pytest.mark.asyncio
async def test_agent_tool_invoke_signature():
  """invoke must accept (ctx, arguments) and return a value."""
  at = AgentTool(agent=Agent(model="test", tools=[]))
  runner = _make_mock_runner()
  session = await runner._create_session(Agent(model="test", tools=[]), deps=None)
  ctx = RunContext(session)

  result = await at.invoke(ctx, {"task": "test"})
  assert result is not None


# --------------------------------------------------------------------------- #
# 10. Child-session lineage
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_agent_tool_resumes_running_child_from_lineage():
  """A running lineage record reuses the orphaned child session."""
  runner = _make_mock_runner(response_content="resumed child answer")
  child_agent = Agent(model="child", tools=[])
  at = AgentTool(agent=child_agent)
  parent = await runner._create_session(Agent(model="parent", tools=[]), deps=None)

  # Simulate an orphaned child session checkpointed mid-run.
  orphan = await runner._create_session(child_agent, deps=None)
  await orphan.memory.add("previous turn marker", MemoryRole.USER)
  orphan.status = SessionStatus.RUNNING
  await runner.checkpoint_store.save_session(orphan.session_id, orphan.to_state())

  prompt = "do something"
  parent.extension_state["agent_tool_lineage"] = {
    at._lineage_key(prompt): {
      "child_session_id": orphan.session_id,
      "status": "running",
      "result_text": None,
    }
  }

  result = await at.invoke(RunContext(parent), {"task": prompt})

  assert result == "resumed child answer"
  # The orphan session was reused: its pre-existing memory reached the LLM.
  calls = runner.llm.chat.call_args_list
  messages = calls[0].kwargs.get("messages") or calls[0][1].get("messages")
  all_content = " ".join(str(m.content) for m in messages)
  assert "previous turn marker" in all_content
  # The lineage record still points at the same child and is now completed.
  record = parent.extension_state["agent_tool_lineage"][at._lineage_key(prompt)]
  assert record["child_session_id"] == orphan.session_id
  assert record["status"] == "completed"
  assert record["result_text"] == "resumed child answer"


@pytest.mark.asyncio
async def test_agent_tool_returns_cached_result_for_completed_lineage():
  """A completed lineage record returns the cached result without running."""
  runner = _make_mock_runner()
  at = AgentTool(agent=Agent(model="child", tools=[]))
  parent = await runner._create_session(Agent(model="parent", tools=[]), deps=None)

  prompt = "do something"
  parent.extension_state["agent_tool_lineage"] = {
    at._lineage_key(prompt): {
      "child_session_id": "old-child",
      "status": "completed",
      "result_text": "cached result",
    }
  }

  result = await at.invoke(RunContext(parent), {"task": prompt})

  assert result == "cached result"
  runner.llm.chat.assert_not_called()
  assert at._lineage_key(prompt) not in parent.extension_state["agent_tool_lineage"]


@pytest.mark.asyncio
async def test_agent_tool_records_lineage_for_new_child_session():
  """Without a lineage record, a fresh child session runs and is recorded."""
  runner = _make_mock_runner(response_content="fresh answer")
  at = AgentTool(agent=Agent(model="child", tools=[]))
  parent = await runner._create_session(Agent(model="parent", tools=[]), deps=None)

  result = await at.invoke(RunContext(parent), {"task": "fresh task"})

  assert result == "fresh answer"
  runner.llm.chat.assert_called()
  record = parent.extension_state["agent_tool_lineage"][at._lineage_key("fresh task")]
  assert record["status"] == "completed"
  assert record["result_text"] == "fresh answer"
  assert record["child_session_id"] != parent.session_id
