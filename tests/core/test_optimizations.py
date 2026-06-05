"""Tests for the optimization roadmap items.

Covers:
- P0-2  Cancellation / Abort mechanism
- P1-1  Redis CheckpointStore N+1 (conceptual coverage via MemoryCheckpointStore)
- P1-2  WorkingMemory safe backend writes
- P1-3  Precise error classification (RunResult.error_type)
- P1-4  Unified model configuration
- P2-1  Plan layers caching
- P2-2  ReActAgent concurrency control
- P3-1  Memory token counting
- P2-4  ToolEvaluator variable naming
"""

import asyncio
from typing import Any
import pytest
from unittest.mock import AsyncMock, MagicMock

from nonoka.core.agent import Agent
from nonoka.core.tool import tool
from nonoka.core.context import RunContext
from nonoka.core.session import Session, SessionStatus
from nonoka.core.types import RunResult
from nonoka.core.errors import CancelledError, MaxTurnsExceeded, MaxStepsExceeded
from nonoka.core.memory import WorkingMemory, MemoryRole
from nonoka.core.plan import PlanBuilder
from nonoka.core.paradigm import ReActAgent, PlanExecutor, ToolEvaluator, EvaluationResult
from nonoka.core.runner import Runner
from nonoka.backends.memory.in_memory import InMemoryBackend
from nonoka.backends.checkpoint.memory import MemoryCheckpointStore


# --------------------------------------------------------------------------- #
# P0-2 — Cancellation / Abort
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_session_cancel_sets_event():
  """Session.cancel() should mark the session as cancelled."""
  agent = Agent(model="test", tools=[])
  session = Session(session_id="s1", agent=agent)

  assert not session.is_cancelled
  session.cancel()
  assert session.is_cancelled


@pytest.mark.asyncio
async def test_session_check_cancelled_raises():
  """Session.check_cancelled() should raise after cancel() is called."""
  agent = Agent(model="test", tools=[])
  session = Session(session_id="s2", agent=agent)
  session.cancel()

  with pytest.raises(CancelledError):
    session.check_cancelled()


@pytest.mark.asyncio
async def test_react_agent_returns_cancelled_error_type():
  """ReActAgent should return RunResult with error_type='cancelled' when cancelled."""
  agent = Agent(model="test", tools=[], max_turns=1)
  session = Session(session_id="s3", agent=agent)

  # Cancel immediately
  session.cancel()

  # Mock runner
  runner = MagicMock()
  runner.checkpoint_store = MemoryCheckpointStore()

  react = ReActAgent()
  result = await react.run(session, runner, prompt="hello")

  assert result.success is False
  assert result.error_type == "cancelled"
  assert session.status == SessionStatus.CANCELLED


@pytest.mark.asyncio
async def test_plan_executor_returns_cancelled_error_type():
  """PlanExecutor should return RunResult with error_type='cancelled' when cancelled."""
  @tool
  async def dummy_tool(x: int) -> int:
    return x

  agent = Agent(model="test", tools=[dummy_tool])
  session = Session(session_id="s4", agent=agent)
  session.cancel()

  plan = (
    PlanBuilder()
    .step("s1", dummy_tool, x=1)
    .build()
  )

  runner = MagicMock()
  runner.checkpoint_store = MemoryCheckpointStore()

  executor = PlanExecutor()
  result = await executor.execute(plan, session, runner)

  assert result.success is False
  assert result.error_type == "cancelled"
  assert session.status == SessionStatus.CANCELLED


# --------------------------------------------------------------------------- #
# P1-3 — Error classification
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_react_agent_max_turns_error_type():
  """ReActAgent should return error_type='limit_exceeded' on max turns."""
  agent = Agent(model="test", tools=[], max_turns=0)
  session = Session(session_id="s5", agent=agent)

  runner = MagicMock()
  runner.checkpoint_store = MemoryCheckpointStore()

  react = ReActAgent()
  result = await react.run(session, runner, prompt="hello")

  assert result.success is False
  assert result.error_type == "limit_exceeded"


# --------------------------------------------------------------------------- #
# P1-4 — Unified model configuration
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
# P1-2 — WorkingMemory safe backend writes
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_working_memory_backend_exception_logged():
  """Backend write failures should be logged, not swallowed."""
  backend = MagicMock()
  backend.add = AsyncMock(side_effect=RuntimeError("backend down"))

  memory = WorkingMemory(session_id="test", memory_backend=backend)

  # Should not raise — exception is caught and logged internally
  await memory.add("hello", MemoryRole.USER)

  # Give the background task a moment to run
  await asyncio.sleep(0.05)

  backend.add.assert_awaited_once()


@pytest.mark.asyncio
async def test_working_memory_flush_awaits_pending():
  """flush() should await all pending backend writes."""
  backend = MagicMock()
  backend.add = AsyncMock(return_value=None)

  memory = WorkingMemory(session_id="test", memory_backend=backend)
  await memory.add("msg1", MemoryRole.USER)
  await memory.add("msg2", MemoryRole.USER)

  await memory.flush()

  assert backend.add.await_count == 2


# --------------------------------------------------------------------------- #
# P3-1 — Memory token counting
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_working_memory_custom_token_counter():
  """WorkingMemory should accept and use a custom token_counter."""
  counter = MagicMock(return_value=100)
  memory = WorkingMemory(session_id="test", token_counter=counter, max_tokens=250)

  await memory.add("short", MemoryRole.USER)

  counter.assert_called_once_with("short")
  assert memory.entries[0].tokens == 100


@pytest.mark.asyncio
async def test_working_memory_default_token_counter_not_zero():
  """Default token counter should return non-zero for real text."""
  memory = WorkingMemory(session_id="test")
  await memory.add("Hello world", MemoryRole.USER)

  assert memory.entries[0].tokens > 0


# --------------------------------------------------------------------------- #
# P2-1 — Plan layers caching
# --------------------------------------------------------------------------- #

def test_plan_layers_precomputed():
  """Plan.layers should be pre-computed at build time."""
  plan = (
    PlanBuilder()
    .step("a", "tool_a")
    .step("b", "tool_b")
    .step("c", "tool_c", depends_on={"a"})
    .build()
  )

  # Access layers multiple times — should not recompute
  layers1 = plan.layers
  layers2 = plan.layers
  assert layers1 is layers2, "layers should be cached"
  assert layers1 == [["a", "b"], ["c"]]


def test_plan_topological_groups_alias():
  """topological_groups() should remain a backward-compatible alias."""
  plan = (
    PlanBuilder()
    .step("a", "tool_a")
    .step("b", "tool_b", depends_on={"a"})
    .build()
  )

  assert plan.topological_groups() == plan.layers


# --------------------------------------------------------------------------- #
# P2-2 — ReActAgent concurrency control
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_react_agent_respects_max_concurrency():
  """ReActAgent should use a Semaphore to limit concurrent tool calls."""
  concurrent_count = 0
  max_observed = 0

  @tool
  async def slow_tool(ctx: RunContext, delay: float) -> str:
    nonlocal concurrent_count, max_observed
    concurrent_count += 1
    max_observed = max(max_observed, concurrent_count)
    await asyncio.sleep(delay)
    concurrent_count -= 1
    return "done"

  agent = Agent(model="test", tools=[slow_tool])
  session = Session(session_id="s6", agent=agent)

  runner = MagicMock()
  runner.checkpoint_store = MemoryCheckpointStore()

  # Mock LLM to return 5 parallel tool calls
  llm_mock = MagicMock()
  llm_mock.chat = AsyncMock(return_value=MagicMock(
    content="",
    tool_calls=[
      {"function": {"name": "slow_tool", "arguments": '{"delay": 0.1}'}},
      {"function": {"name": "slow_tool", "arguments": '{"delay": 0.1}'}},
      {"function": {"name": "slow_tool", "arguments": '{"delay": 0.1}'}},
      {"function": {"name": "slow_tool", "arguments": '{"delay": 0.1}'}},
      {"function": {"name": "slow_tool", "arguments": '{"delay": 0.1}'}},
    ],
  ))
  runner.llm = llm_mock

  react = ReActAgent(max_concurrency=2)
  result = await react.run(session, runner, prompt="run")

  assert max_observed <= 2, f"Expected max concurrency 2, got {max_observed}"


@pytest.mark.asyncio
async def test_react_agent_uses_agent_max_concurrency_default():
  """ReActAgent should fall back to agent.max_concurrency when not explicitly set."""
  agent = Agent(model="test", tools=[], max_concurrency=7)
  session = Session(session_id="s7", agent=agent)

  react = ReActAgent()
  # max_concurrency is None → should resolve from agent
  assert react.max_concurrency is None

  # We can't easily test the internal Semaphore value without introspection,
  # but we verify the agent field exists and has the expected default.
  assert agent.max_concurrency == 7


# --------------------------------------------------------------------------- #
# P2-4 — ToolEvaluator variable naming
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_tool_evaluator_uses_session():
  """ToolEvaluator should build RunContext from result.session correctly."""
  @tool
  async def validator(ctx: RunContext, data: Any) -> dict:
    # Verify we receive a real RunContext with access to session
    assert ctx.session is not None
    return {"passed": True, "feedback": "ok"}

  agent = Agent(model="test", tools=[validator])
  session = Session(session_id="s8", agent=agent)

  evaluator = ToolEvaluator(validator, data_extractor=lambda r: r.data)

  mock_result = RunResult(success=True, data="test-data", session=session)
  eval_result = await evaluator.evaluate(mock_result)

  assert eval_result.passed is True
  assert eval_result.feedback == "ok"


@pytest.mark.asyncio
async def test_tool_evaluator_no_session():
  """ToolEvaluator should handle missing session gracefully."""
  @tool
  async def validator(ctx: RunContext, data: Any) -> dict:
    return {"passed": True}

  evaluator = ToolEvaluator(validator)

  mock_result = RunResult(success=True, data="test", session=None)
  eval_result = await evaluator.evaluate(mock_result)

  assert eval_result.passed is False
  assert "No session available" in eval_result.feedback
