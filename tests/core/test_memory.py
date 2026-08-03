import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from nonoka.core.memory import (
  MemoryEntry,
  WorkingMemory,
  MemoryRole,
  microcompact_superseded_tool_results,
)
from nonoka.core.runtime import RuntimeLimits
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
async def test_protocol_compactor_keeps_evidence_ledger_and_valid_pairs():
  memory = WorkingMemory(session_id="ledger", max_tokens=1000, token_counter=len)
  await memory.add("implement the task", MemoryRole.USER)
  await memory.add(
    "", MemoryRole.ASSISTANT, defer_budget=True,
    tool_calls=[{
      "id": "call-1",
      "function": {"name": "bash", "arguments": '{"command":"pytest -q"}'},
    }],
  )
  await memory.add(
    "x" * 2000, MemoryRole.TOOL, defer_budget=True,
    tool_call_id="call-1", tool_name="bash", exit_code=1,
    artifact_ref="trace://bash-call-1.txt",
  )

  metrics = await memory.enforce_budget(RuntimeLimits(
    max_context_bytes=1600, max_context_tokens=1000, max_tool_messages=1,
  ))

  assert metrics is not None
  assert not [entry for entry in memory.entries if entry.role == MemoryRole.TOOL]
  ledger = next(entry for entry in memory.entries if entry.metadata.get("evidence_ledger"))
  assert "pytest -q" in ledger.content
  assert "trace://bash-call-1.txt" in ledger.content
  assert "implement the task" in [entry.content for entry in memory.entries]


@pytest.mark.asyncio
async def test_protocol_compactor_preserves_activated_skill_tool_result():
  memory = WorkingMemory(session_id="skill-context", max_tokens=10_000, token_counter=len)
  await memory.add("use the skill", MemoryRole.USER)
  await memory.add(
    "", MemoryRole.ASSISTANT, defer_budget=True,
    tool_calls=[{
      "id": "skill-call",
      "function": {"name": "load_skill", "arguments": '{"name":"review"}'},
    }],
  )
  await memory.add(
    "protected skill instructions", MemoryRole.TOOL, defer_budget=True,
    tool_call_id="skill-call", tool_name="load_skill", context_protected=True,
  )
  await memory.add("old disposable result" * 100, MemoryRole.USER, defer_budget=True)
  await memory.add("continue", MemoryRole.USER, defer_budget=True)

  await memory.enforce_budget(RuntimeLimits(
    max_context_bytes=900, max_context_tokens=10_000,
  ))

  contents = [entry.content for entry in memory.entries]
  assert "protected skill instructions" in contents
  assert any(
    entry.role == MemoryRole.ASSISTANT
    and entry.metadata.get("tool_calls", [{}])[0].get("id") == "skill-call"
    for entry in memory.entries
  )


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


# --------------------------------------------------------------------------- #
# Microcompaction — superseded tool results
# --------------------------------------------------------------------------- #

def _read_call(call_id: str, path: str) -> dict:
  return {
    "id": call_id,
    "function": {"name": "Read", "arguments": json.dumps({"file_path": path})},
  }


def _read_entries(path_a: str, path_b: str) -> list[MemoryEntry]:
  return [
    MemoryEntry(role=MemoryRole.USER, content="task", tokens=4),
    MemoryEntry(
      role=MemoryRole.ASSISTANT, content="", tokens=1,
      metadata={"tool_calls": [_read_call("c1", path_a)]},
    ),
    MemoryEntry(
      role=MemoryRole.TOOL, content="old-content", tokens=11,
      metadata={"tool_call_id": "c1"},
    ),
    MemoryEntry(
      role=MemoryRole.ASSISTANT, content="", tokens=1,
      metadata={"tool_calls": [_read_call("c2", path_b)]},
    ),
    MemoryEntry(
      role=MemoryRole.TOOL, content="new-content", tokens=11,
      metadata={"tool_call_id": "c2"},
    ),
  ]


def test_microcompact_supersedes_duplicate_read_results():
  """Only the newest result for the same logical call keeps its content."""
  entries = _read_entries("/a.py", "/a.py")
  result = microcompact_superseded_tool_results(entries, len)

  # Entries are never removed — assistant/tool protocol units stay intact.
  assert len(result) == len(entries)
  old, new = result[2], result[4]
  assert old.content == "[superseded by newer read result]"
  assert old.metadata["superseded"] is True
  assert old.tokens == len(old.content)
  assert new.content == "new-content"
  assert "superseded" not in new.metadata


def test_microcompact_keeps_results_for_different_paths():
  entries = _read_entries("/a.py", "/b.py")
  result = microcompact_superseded_tool_results(entries, len)

  assert result[2].content == "old-content"
  assert result[4].content == "new-content"


def test_microcompact_skips_entries_without_reliable_arguments():
  """Results whose call arguments cannot be determined are left untouched."""
  entries = [
    MemoryEntry(role=MemoryRole.USER, content="task", tokens=4),
    MemoryEntry(
      role=MemoryRole.TOOL, content="first", tokens=5,
      metadata={"tool_call_id": "missing-call"},
    ),
    MemoryEntry(
      role=MemoryRole.TOOL, content="second", tokens=6,
      metadata={"tool_call_id": "missing-call"},
    ),
  ]
  result = microcompact_superseded_tool_results(entries, len)

  assert [entry.content for entry in result] == ["task", "first", "second"]


@pytest.mark.asyncio
async def test_enforce_budget_microcompacts_before_compacting():
  """A superseded duplicate result can bring the window back under budget."""
  memory = WorkingMemory(
    session_id="micro", max_tokens=100, token_counter=len,
    reserve_output_tokens=10, compaction_buffer_tokens=5,
  )
  await memory.add("do the task", MemoryRole.USER)
  await memory.add(
    "", MemoryRole.ASSISTANT, defer_budget=True,
    tool_calls=[_read_call("c1", "/a.py")],
  )
  await memory.add("r" * 50, MemoryRole.TOOL, defer_budget=True, tool_call_id="c1")
  await memory.add(
    "", MemoryRole.ASSISTANT, defer_budget=True,
    tool_calls=[_read_call("c2", "/a.py")],
  )
  await memory.add("r" * 50, MemoryRole.TOOL, defer_budget=True, tool_call_id="c2")
  await memory.enforce_budget()

  # Microcompaction alone got under the trigger: nothing was evicted and no
  # evidence ledger had to be created.
  assert len(memory.entries) == 5
  contents = [entry.content for entry in memory.entries]
  assert "[superseded by newer read result]" in contents
  assert not any(entry.metadata.get("evidence_ledger") for entry in memory.entries)


# --------------------------------------------------------------------------- #
# Early trigger threshold and output reserve
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_enforce_budget_triggers_before_cap_and_reserves_output():
  """Compaction starts at ``max_tokens - buffer`` and trims to ``- reserve``."""
  memory = WorkingMemory(
    session_id="threshold", max_tokens=100, token_counter=len,
    reserve_output_tokens=20, compaction_buffer_tokens=10,
  )
  await memory.add("s" * 50, MemoryRole.SYSTEM)
  # Total is 94 — under the raw cap (100) but over the trigger (90).
  await memory.add("u" * 20, MemoryRole.USER, defer_budget=True)
  await memory.add("a" * 24, MemoryRole.ASSISTANT)

  total = sum(entry.tokens for entry in memory.entries)
  assert total <= 80  # max_tokens - reserve_output_tokens


@pytest.mark.asyncio
async def test_enforce_budget_no_trigger_below_buffer_threshold():
  memory = WorkingMemory(
    session_id="below-trigger", max_tokens=100, token_counter=len,
    reserve_output_tokens=20, compaction_buffer_tokens=10,
  )
  await memory.add("s" * 50, MemoryRole.SYSTEM)
  await memory.add("u" * 20, MemoryRole.USER, defer_budget=True)
  # Total is 90 — exactly at the trigger, so nothing is compacted.
  await memory.add("a" * 20, MemoryRole.ASSISTANT)

  assert [entry.content for entry in memory.entries] == [
    "s" * 50, "u" * 20, "a" * 20,
  ]


# --------------------------------------------------------------------------- #
# Summary layer — circuit breaker and widened range
# --------------------------------------------------------------------------- #

class FailingLLMProvider:
  """Summary LLM that always raises, for circuit-breaker tests."""

  def __init__(self):
    self.call_count = 0

  async def chat(self, messages, **kwargs):
    self.call_count += 1
    raise RuntimeError("summary backend down")


@pytest.mark.asyncio
async def test_summary_circuit_breaker_falls_back_to_ledger_compaction():
  """Three consecutive summary failures disable summarisation for the session."""
  llm = FailingLLMProvider()
  memory = WorkingMemory(
    session_id="breaker", max_tokens=100, token_counter=len,
    summary_llm=llm, reserve_output_tokens=10, compaction_buffer_tokens=0,
  )
  # Spy on the deterministic compactor to observe summary fallbacks. (The
  # ledger entry itself may legitimately be dropped again when it does not
  # fit the budget — that is pre-existing compactor behavior.)
  compactor_spy = AsyncMock(wraps=memory.context_compactor.compact)
  memory.context_compactor.compact = compactor_spy

  async def overflow(index: int) -> None:
    for suffix in ("a", "b"):
      await memory.add(f"user-{index}-{suffix}", MemoryRole.USER, defer_budget=True)
      await memory.add("x" * 200, MemoryRole.ASSISTANT, defer_budget=True)
    await memory.enforce_budget()

  for round_index in range(4):
    await overflow(round_index)

  # The summariser was attempted exactly three times, then the breaker opened.
  assert llm.call_count == 3
  assert memory._summary_disabled is True
  # Every failure fell back to the deterministic ledger compactor, and the
  # fourth overflow used it directly without calling the summariser.
  assert compactor_spy.await_count == 4


@pytest.mark.asyncio
async def test_summary_covers_oldest_third_of_chat_history():
  """The summary range widened from the oldest ~5 to the oldest ~1/3."""
  mock_llm = MockLLMProvider()
  memory = WorkingMemory(
    session_id="summary-range", max_tokens=200, token_counter=len,
    summary_llm=mock_llm, reserve_output_tokens=10, compaction_buffer_tokens=0,
  )
  await memory.add("sys", MemoryRole.SYSTEM)
  for i in range(24):
    await memory.add(f"message-{i:02d} " + "x" * 40, MemoryRole.USER, defer_budget=True)
  await memory.enforce_budget()

  assert mock_llm.call_count == 1
  prompt = mock_llm.last_messages[0].content
  # 24 chat entries → oldest 8 summarised; the old ~5 cap would have stopped
  # before message-05.
  assert "message-00" in prompt
  assert "message-05" in prompt
  assert "message-07" in prompt
  assert "message-08" not in prompt


@pytest.mark.live
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

  model_name = os.getenv("NONOKA_TEST_MODEL", "deepseek-v4-pro")
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
