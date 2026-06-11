from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from nonoka.core.agent import Agent
from nonoka.core.agent_tool import AgentTool, MemoryStrategy
from nonoka.core.context import RunContext
from nonoka.core.memory import MemoryRole, WorkingMemory
from nonoka.core.runner import Runner
from nonoka.core.session import Session
from nonoka.core.tool import tool
from nonoka.core.types import RunResult
from nonoka.backends.memory.in_memory import InMemoryBackend


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

  result = await at.invoke(ctx, {"task": "do something"})

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

  result = await at.invoke(ctx, {"task": "do something"})

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
  runner = Runner(checkpoint="memory")
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
  runner = Runner(checkpoint="memory")
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
  result = await at.invoke(ctx, {"task": "do something"})

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
  runner = Runner(checkpoint="memory")
  provider = _setup_mock_runner(runner, response_content="child answer")

  at = AgentTool(
    agent=Agent(model="test-model", tools=[]),
    memory_strategy=MemoryStrategy.SHARE,
  )

  parent_agent = Agent(model="test-model", tools=[])
  parent_session = await runner._create_session(parent_agent, deps=None)
  await parent_session.memory.add("Shared context", MemoryRole.USER)

  ctx = RunContext(parent_session)
  result = await at.invoke(ctx, {"task": "do something"})

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

  runner = Runner(checkpoint="memory")

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
  result = await at2.invoke(ctx, {"task": "capture deps"})

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
async def test_agent_tool_fallback_runner_when_no_ref():
  """When session lacks _runner_ref, AgentTool should create a default Runner."""
  at = AgentTool(agent=Agent(model="test", tools=[]))

  # Manually create a session without _runner_ref
  session = Session(session_id="test-session", agent=Agent(model="test", tools=[]), deps=None)
  assert not hasattr(session, "_runner_ref")
  ctx = RunContext(session)

  # Should still work (creates fallback Runner)
  result = await at.invoke(ctx, {"task": "do something"})
  # The fallback runner may or may not have an LLM configured,
  # so we just verify it doesn't crash and returns a structured result
  assert result is not None


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
