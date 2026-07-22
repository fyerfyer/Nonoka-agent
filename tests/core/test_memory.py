import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from nonoka.core.memory import WorkingMemory, MemoryRole
from nonoka.backends.memory.in_memory import InMemoryBackend
from nonoka.core.llm import LLMResponse


class MockLLMProvider:
  """Mock LLM Provider for testing summary strategy without real API calls."""

  def __init__(self):
    self.call_count = 0
    self.last_messages = []

  async def chat(self, messages, **kwargs):
    self.call_count += 1
    self.last_messages = messages
    return LLMResponse(content="[MOCKED SUMMARY]", usage={"total_tokens": 10})

  def count_tokens(self, content):
    if isinstance(content, list):
      return sum(len(str(m)) // 3 for m in content)
    return len(str(content)) // 3 if content else 0


# --------------------------------------------------------------------------- #
# Safe backend writes
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
# Token counting
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
# Budget strategies
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_working_memory_sliding_window():
  """Default budget strategy (sliding window) evicts oldest non-system entries."""
  memory = WorkingMemory(session_id="test-1", max_tokens=15)

  # Add System prompt (tokens ~ 9)
  await memory.add("You are a helpful assistant", MemoryRole.SYSTEM)

  # Add user messages (tokens ~ 2 each)
  await memory.add("Hello 1", MemoryRole.USER)
  await memory.add("Hello 2", MemoryRole.USER)
  await memory.add("Hello 3", MemoryRole.USER)
  await memory.add("Hello 4", MemoryRole.USER)
  await memory.add("Hello 5", MemoryRole.USER)

  context = await memory.get_context()

  # With max_tokens=15, it should evict older messages but KEEP the SYSTEM prompt
  assert context[0].role == MemoryRole.SYSTEM
  assert "You are a helpful assistant" in context[0].content

  # Verify the oldest user messages are gone
  chat_contents = [e.content for e in context if e.role != MemoryRole.SYSTEM]
  assert "Hello 1" not in chat_contents
  assert "Hello 5" in chat_contents


@pytest.mark.asyncio
async def test_working_memory_evicts_a_complete_tool_call_batch():
  """Context trimming must never leave a provider-invalid orphan tool result."""
  memory = WorkingMemory(
    session_id="tool-batch", max_tokens=10, token_counter=len,
  )
  tool_calls = [
    {"id": "call-1", "function": {"name": "inspect", "arguments": "{}"}},
    {"id": "call-2", "function": {"name": "inspect", "arguments": "{}"}},
  ]

  await memory.add("user", MemoryRole.USER)
  await memory.add("calls", MemoryRole.ASSISTANT, tool_calls=tool_calls)
  # A ReAct tool batch is appended atomically for memory-budget purposes.
  await memory.add("first-result", MemoryRole.TOOL, defer_budget=True, tool_call_id="call-1")
  await memory.add("second-result", MemoryRole.TOOL, defer_budget=True, tool_call_id="call-2")
  await memory.enforce_budget()

  assert [entry.content for entry in memory.entries if entry.role == MemoryRole.USER] == ["user"]
  assert not [entry for entry in memory.entries if entry.role == MemoryRole.TOOL]
  assert not [entry for entry in memory.entries if entry.role == MemoryRole.ASSISTANT]


@pytest.mark.asyncio
async def test_working_memory_summary_strategy():
  """When summary_llm is provided, WorkingMemory auto-summarises old chats."""
  mock_llm = MockLLMProvider()
  memory = WorkingMemory(session_id="test-2", max_tokens=15, summary_llm=mock_llm)

  await memory.add("System prompt", MemoryRole.SYSTEM)

  # Add enough chats to trigger summary
  for i in range(1, 8):
    await memory.add(f"Message {i} content", MemoryRole.USER)

  context = await memory.get_context()

  # LLM should have been called for summarization
  assert mock_llm.call_count > 0

  # The context should now contain a SYSTEM message with the summary
  system_entries = [e for e in context if e.role == MemoryRole.SYSTEM]
  assert any("[MOCKED SUMMARY]" in e.content for e in system_entries)


@pytest.mark.asyncio
async def test_working_memory_rag_integration():
  """WorkingMemory retrieves from MemoryBackend and injects into context."""
  backend = InMemoryBackend()
  await backend.add("User's favorite color is blue", session_id="test-3")

  memory = WorkingMemory(session_id="test-3", memory_backend=backend)

  await memory.add("System prompt", MemoryRole.SYSTEM)
  await memory.add("favorite color", MemoryRole.USER)

  context = await memory.get_context()

  # It should have injected the retrieved memory into the context as a SYSTEM prompt
  system_entries = [e for e in context if e.role == MemoryRole.SYSTEM]

  assert len(system_entries) == 2  # Original System + RAG System
  assert any("blue" in e.content for e in system_entries)


@pytest.mark.asyncio
async def test_working_memory_summary_strategy_real_llm():
  """Test summarisation with a real LLM (requires OPENAI_API_KEY)."""
  import os
  from dotenv import load_dotenv
  from nonoka.core.llm import LiteLLMProvider

  load_dotenv()

  api_key = os.getenv("OPENAI_API_KEY")
  base_url = os.getenv("OPENAI_BASE_URL")
  if not api_key:
    pytest.skip("No OPENAI_API_KEY found, skipping real LLM test for memory.")

  model_name = "deepseek-chat"
  if base_url:
    model_name = f"openai/{model_name}"

  real_llm = LiteLLMProvider(model=model_name, api_key=api_key, base_url=base_url)

  # Set a small budget to easily trigger summarization
  memory = WorkingMemory(session_id="test-real", max_tokens=60, summary_llm=real_llm)

  await memory.add("System prompt: You are a smart assistant.", MemoryRole.SYSTEM)

  for i in range(1, 8):
    # We add some substantial text so the token count triggers the budget limit quickly
    await memory.add(
      f"This is user message {i}. The user is discussing some topic.",
      MemoryRole.USER,
    )

  context = await memory.get_context()

  system_entries = [e for e in context if e.role == MemoryRole.SYSTEM]
  summary_entries = [e for e in system_entries if "History Summary:" in e.content]

  assert len(summary_entries) > 0, "Summary entry not found in context"
  print(f"\n[Real Summary]: {summary_entries[0].content}")
