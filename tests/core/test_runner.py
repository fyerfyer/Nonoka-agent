"""Tests for the Runner orchestrator."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from nonoka.core.agent import Agent
from nonoka.core.tool import tool
from nonoka.core.types import RetryPolicy
from nonoka.core.runner import Runner, StreamEvent
from nonoka.core.llm import LLMResponse, LLMStreamChunk
from nonoka.backends.checkpoint.memory import MemoryCheckpointStore


# --------------------------------------------------------------------------- #
# Unified model configuration
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_runner_no_model_in_constructor():
  """Runner should not accept 'model' in __init__ anymore."""
  with pytest.raises(TypeError):
    Runner(model="gpt-4")  # type: ignore[call-arg]


def test_runner_caches_llm_per_model():
  """Runner._ensure_llm should cache providers per model."""
  runner = Runner()
  agent_a = Agent(model="model-a", tools=[])
  agent_b = Agent(model="model-b", tools=[])

  llm_a1 = runner._ensure_llm(agent_a)
  llm_a2 = runner._ensure_llm(agent_a)
  llm_b1 = runner._ensure_llm(agent_b)

  assert llm_a1 is llm_a2, "Same model should return cached provider"
  assert llm_a1 is not llm_b1, "Different models should create different providers"


# --------------------------------------------------------------------------- #
# Retry / timeout policy propagation
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_runner_passes_retry_policy_to_llm():
  """Runner should create LiteLLMProvider with agent retry/timeout config."""
  runner = Runner()
  agent = Agent(
    model="test-model",
    tools=[],
    default_retry=RetryPolicy(max_retries=7, backoff=1.5),
    default_timeout=42.0,
  )
  provider = runner._create_llm(agent)

  assert provider.retry_policy.max_retries == 7
  assert provider.retry_policy.backoff == 1.5
  assert provider.timeout == 42.0


# --------------------------------------------------------------------------- #
# Streaming interface
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_runner_run_react_stream_returns_events():
  """Runner.run_react_stream should yield StreamEvents."""
  agent = Agent(model="test", tools=[], max_turns=1, temperature=0.2, max_tokens=321)

  runner = Runner()

  captured = {}

  async def fake_stream(*args, **kwargs):
    captured.update(kwargs)
    yield LLMStreamChunk(finish_reason="stop")

  provider = MagicMock()
  provider.chat_stream = fake_stream
  provider.chat = AsyncMock(return_value=MagicMock(content="ok", tool_calls=None, usage={}))

  runner._create_llm = lambda agent: provider  # type: ignore[method-assign]

  events = []
  async for event in runner.run_react_stream(agent, prompt="hello", deps=None):
    events.append(event)

  assert len(events) >= 1
  assert events[-1].type == "final"
  assert captured["temperature"] == 0.2
  assert captured["max_tokens"] == 321


@pytest.mark.asyncio
async def test_runner_forwards_agent_generation_config_to_react_provider():
  agent = Agent(model="test", tools=[], max_turns=1, temperature=0.0, max_tokens=123)
  runner = Runner()
  provider = MagicMock()
  provider.chat = AsyncMock(return_value=LLMResponse(content="ok"))
  runner._create_llm = lambda _agent: provider  # type: ignore[method-assign]

  result = await runner.run_react(agent, prompt="hello", deps=None)

  assert result.success
  provider.chat.assert_awaited_once()
  assert provider.chat.call_args.kwargs["temperature"] == 0.0
  assert provider.chat.call_args.kwargs["max_tokens"] == 123


# --------------------------------------------------------------------------- #
# Session memory recovery on resume (Bug fix: P0)
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_create_session_restores_memory_entries_from_checkpoint():
  """When session_id exists in checkpoint store, memory entries should be restored."""
  runner = Runner()
  agent = Agent(model="test", tools=[])

  # First, create a session and populate its memory
  session1 = await runner._create_session(agent, deps=None)
  assert session1.memory is not None
  from nonoka.core.memory import MemoryRole
  await session1.memory.add("Hello", MemoryRole.USER)
  await session1.memory.add("Hi there", MemoryRole.ASSISTANT)

  # Save to checkpoint
  await runner.checkpoint_store.save_session(session1.session_id, session1.to_state())

  # Now create a new session with the same ID — memory should be restored
  session2 = await runner._create_session(agent, deps=None, session_id=session1.session_id)
  assert session2.memory is not None
  assert len(session2.memory.entries) == 2
  assert session2.memory.entries[0].content == "Hello"
  assert session2.memory.entries[0].role == MemoryRole.USER
  assert session2.memory.entries[1].content == "Hi there"
  assert session2.memory.entries[1].role == MemoryRole.ASSISTANT


@pytest.mark.asyncio
async def test_create_session_with_unknown_session_id_creates_fresh_memory():
  """When session_id is not found in checkpoint, a fresh WorkingMemory is created."""
  runner = Runner()
  agent = Agent(model="test", tools=[])

  session = await runner._create_session(agent, deps=None, session_id="never-seen-before")
  assert session.memory is not None
  assert len(session.memory.entries) == 0


@pytest.mark.asyncio
async def test_create_session_restores_memory_entries_from_checkpoint_with_role():
  """Memory entries with different roles are restored correctly."""
  runner = Runner()
  agent = Agent(model="test", tools=[])

  session1 = await runner._create_session(agent, deps=None)
  from nonoka.core.memory import MemoryRole
  await session1.memory.add("System prompt", MemoryRole.SYSTEM)
  await session1.memory.add("User query", MemoryRole.USER)
  await session1.memory.add("Tool result", MemoryRole.TOOL)

  await runner.checkpoint_store.save_session(session1.session_id, session1.to_state())

  session2 = await runner._create_session(agent, deps=None, session_id=session1.session_id)
  assert len(session2.memory.entries) == 3
  roles = [e.role for e in session2.memory.entries]
  assert roles == [MemoryRole.SYSTEM, MemoryRole.USER, MemoryRole.TOOL]


@pytest.mark.asyncio
async def test_resume_restores_memory_entries():
  """Resume() should restore memory entries from checkpoint."""
  from nonoka.backends.memory.in_memory import InMemoryBackend
  runner = Runner(memory=InMemoryBackend())
  agent = Agent(model="test", tools=[])

  session1 = await runner._create_session(agent, deps=None)
  from nonoka.core.memory import MemoryRole
  await session1.memory.add("Previous context", MemoryRole.USER)
  await session1.memory.add("Assistant response", MemoryRole.ASSISTANT)

  # Manually set status to PAUSED so resume doesn't early-return
  from nonoka.core.session import SessionStatus
  session1.status = SessionStatus.PAUSED
  await runner.checkpoint_store.save_session(session1.session_id, session1.to_state())

  # Resume — resume() needs a memory_backend to create WorkingMemory
  result = await runner.resume(agent, session_id=session1.session_id, deps=None)
  assert result.session is not None
  assert result.session.memory is not None
  assert len(result.session.memory.entries) == 2
  assert result.session.memory.entries[0].content == "Previous context"
  assert result.session.memory.entries[1].content == "Assistant response"
