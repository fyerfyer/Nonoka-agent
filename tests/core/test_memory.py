import pytest
from nonoka.core.memory import WorkingMemory, MemoryRole, DefaultBudgetStrategy, SummaryBudgetStrategy
from nonoka.backends.memory.in_memory import InMemoryBackend
from nonoka.core.llm import LLMResponse


class MockLLMProvider:
  """Mock LLM Provider for testing Summary Strategy without making real API calls"""
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


@pytest.mark.asyncio
async def test_working_memory_sliding_window():
  """Test DefaultBudgetStrategy (Sliding Window) eviction logic"""
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
async def test_working_memory_summary_strategy():
  """Test SummaryBudgetStrategy automatically triggering summarization"""
  mock_llm = MockLLMProvider()
  # Passing summary_llm should auto-route to SummaryBudgetStrategy
  memory = WorkingMemory(session_id="test-2", max_tokens=15, summary_llm=mock_llm)

  assert isinstance(memory.strategy, SummaryBudgetStrategy)

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
  """Test WorkingMemory automatically retrieving from MemoryBackend"""
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
  """Test SummaryBudgetStrategy with real LLM to verify token calculation and summarization"""
  import os
  from dotenv import load_dotenv
  from nonoka.backends.llm.litellm_ext import LiteLLMProvider
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
    await memory.add(f"This is user message {i}. The user is discussing some topic.", MemoryRole.USER)

  context = await memory.get_context()

  system_entries = [e for e in context if e.role == MemoryRole.SYSTEM]
  summary_entries = [e for e in system_entries if "History Summary:" in e.content]

  assert len(summary_entries) > 0, "Summary entry not found in context"
  print(f"\n[Real Summary]: {summary_entries[0].content}")