"""Tests for the Runner orchestrator."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from nonoka.core.agent import Agent
from nonoka.core.tool import tool
from nonoka.core.types import RetryPolicy
from nonoka.core.runner import Runner, StreamEvent
from nonoka.core.llm import LLMStreamChunk
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
  agent = Agent(model="test", tools=[], max_turns=1)

  runner = Runner()

  async def fake_stream(*args, **kwargs):
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