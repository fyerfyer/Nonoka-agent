import os
import pytest
from dotenv import load_dotenv

from nonoka.core.agent import Agent
from nonoka.core.runner import Runner
from nonoka.core.tool import tool
from nonoka.backends.checkpoint.sqlite import SQLiteCheckpointStore
from nonoka.backends.memory.sqlite import SQLiteMemoryBackend

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")


@pytest.fixture
def deepseek_model():
  """Return the DeepSeek model name, skipping if no API key."""
  if not API_KEY:
    pytest.skip("No OPENAI_API_KEY found, skipping real LLM integration test.")
  model = "deepseek-chat"
  if BASE_URL:
    model = f"openai/{model}"
  return model


# --------------------------------------------------------------------------- #
# Real LLM integration tests
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_react_with_sqlite_checkpoint(deepseek_model):
  """ReAct execution with SQLite checkpoint should save and restore session."""

  @tool
  async def get_weather(city: str) -> str:
    return f"Sunny in {city}!"

  agent = Agent(
    model=deepseek_model,
    tools=[get_weather],
    system_prompt="You are a helpful assistant. Use tools when needed.",
    max_turns=3,
  )

  # Use SQLite backends
  checkpoint_store = SQLiteCheckpointStore(":memory:")
  memory_backend = SQLiteMemoryBackend(":memory:")
  runner = Runner(checkpoint=checkpoint_store, memory=memory_backend)

  # Run a simple task
  result = await runner.run_react(agent, "What's the weather in Beijing?", deps=None)

  # Verify session was saved to checkpoint
  assert result.session is not None
  session_id = result.session.session_id

  saved_state = await checkpoint_store.load_session(session_id)
  assert saved_state is not None
  assert saved_state.session_id == session_id
  assert saved_state.turn_count > 0

  await checkpoint_store.close()
  await memory_backend.close()


@pytest.mark.asyncio
async def test_react_memory_persistence_with_sqlite(deepseek_model):
  """SQLite memory backend should preserve conversation history across turns."""

  agent = Agent(
    model=deepseek_model,
    tools=[],
    system_prompt="You are a helpful assistant. Keep responses brief.",
    max_turns=3,
  )

  memory_backend = SQLiteMemoryBackend(":memory:")
  runner = Runner(checkpoint="memory", memory=memory_backend)

  # First turn
  result1 = await runner.run_react(agent, "My name is Alice.", deps=None)
  assert result1.success
  session_id = result1.session.session_id

  # Add to memory
  await memory_backend.add("User's name is Alice", session_id=session_id)

  # Second turn - resume same session
  result2 = await runner.run_react(
    agent, "What's my name?", deps=None, session_id=session_id
  )
  assert result2.success

  # Memory should contain the user name
  history = await memory_backend.get_history(session_id)
  assert len(history) >= 1

  # Search should find the name
  results = await memory_backend.search("Alice", session_id=session_id)
  assert len(results) >= 1

  await memory_backend.close()


@pytest.mark.asyncio
async def test_react_stream_with_sqlite_checkpoint(deepseek_model):
  """Streaming ReAct with SQLite checkpoint should emit correct events."""

  agent = Agent(
    model=deepseek_model,
    tools=[],
    system_prompt="You are a helpful assistant.",
    max_turns=2,
  )

  checkpoint_store = SQLiteCheckpointStore(":memory:")
  runner = Runner(checkpoint=checkpoint_store, memory=None)

  events = []
  async for event in runner.run_react_stream(agent, "Say hello in one word.", deps=None):
    events.append(event)

  # Should have at least content_delta and final events
  assert len(events) >= 2
  assert events[-1].type == "final"
  assert events[-1].data.get("success") is True

  await checkpoint_store.close()


@pytest.mark.asyncio
async def test_plan_execution_with_sqlite_checkpoint(deepseek_model):
  """Plan execution should work with SQLite checkpoint store."""
  from nonoka.core.plan import Plan, Step

  @tool
  async def add(a: int, b: int) -> int:
    return a + b

  agent = Agent(model=deepseek_model, tools=[add], max_steps=5)
  checkpoint_store = SQLiteCheckpointStore(":memory:")
  runner = Runner(checkpoint=checkpoint_store, memory=None)

  plan = Plan(
    steps=[
      Step(id="step-1", tool="add", args={"a": 1, "b": 2}),
    ]
  )

  result = await runner.run_plan(agent, plan, deps=None)
  assert result.success

  # Checkpoint should have saved step result
  saved_state = await checkpoint_store.load_session(result.session.session_id)
  assert saved_state is not None
  assert "step-1" in saved_state.completed_steps
  # Tool results are now normalised to standard shape
  assert saved_state.completed_steps["step-1"].data == {"result": 3, "has_more": False}

  await checkpoint_store.close()
