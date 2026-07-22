from __future__ import annotations

import asyncio
from enum import Enum
from typing import Any, Protocol, runtime_checkable
from pydantic import BaseModel, Field

from nonoka.core.logger import get_logger

_logger = get_logger("nonoka.memory")


class MemoryRole(str, Enum):
  SYSTEM = "system"
  USER = "user"
  ASSISTANT = "assistant"
  TOOL = "tool"


class MemoryEntry(BaseModel):
  role: MemoryRole
  content: str
  metadata: dict[str, Any] = Field(default_factory=dict)
  tokens: int = 0  # Token count


@runtime_checkable
class MemoryBackend(Protocol):
  """Persistent memory storage interface."""

  async def add(
    self, content: str,
    session_id: str | None = None,
    user_id: str | None = None,
    metadata: dict[str, Any] | None = None
  ) -> None: ...

  async def search(
    self, query: str,
    session_id: str | None = None,
    user_id: str | None = None,
    limit: int = 5
  ) -> list[MemoryEntry]: ...

  async def get_history(self, session_id: str, limit: int | None = None) -> list[MemoryEntry]: ...

  async def get_user_memory(self, user_id: str, limit: int = 10) -> list[MemoryEntry]: ...


# --------------------------------------------------------------------------- #
# Token counting
# --------------------------------------------------------------------------- #

def _default_count_tokens(content: str) -> int:
  """Default token counter — uses litellm when available, falls back to a
  UTF-8-aware heuristic that is significantly more accurate than ``len // 3``.
  """
  if not content:
    return 0
  try:
    import litellm
    return litellm.token_counter(model="gpt-4", text=content)
  except Exception:
    # Fallback: ~1 token per UTF-8 byte for CJK, ~0.25 for ASCII.
    # This is still a heuristic but far better than char-count // 3.
    return max(1, len(content.encode("utf-8")) // 3)


def _pop_protocol_unit(entries: list[MemoryEntry], start: int = 0) -> list[MemoryEntry]:
  """Remove one chat exchange without orphaning tool results.

  ``entries`` excludes system messages and preserves conversational order.
  A tool result can only be sent to a provider after the assistant message
  that declared its ID, so an assistant tool-call message and its contiguous
  results must be trimmed together.
  """
  first = entries.pop(start)
  removed = [first]
  if first.role != MemoryRole.ASSISTANT:
    return removed

  raw_calls = first.metadata.get("tool_calls")
  if not isinstance(raw_calls, list):
    return removed
  call_ids = {
    str(call.get("id") or call.get("tool_call_id"))
    for call in raw_calls
    if isinstance(call, dict) and (call.get("id") or call.get("tool_call_id"))
  }
  while start < len(entries) and entries[start].role == MemoryRole.TOOL:
    next_entry = entries[start]
    tool_call_id = next_entry.metadata.get("tool_call_id")
    if call_ids and str(tool_call_id) not in call_ids:
      break
    removed.append(entries.pop(start))
  return removed


def _complete_protocol_prefix_length(entries: list[MemoryEntry], target: int) -> int:
  """Round a history prefix up to complete assistant/tool exchanges."""
  index = 0
  while index < target and index < len(entries):
    entry = entries[index]
    index += 1
    if entry.role != MemoryRole.ASSISTANT:
      continue
    raw_calls = entry.metadata.get("tool_calls")
    if not isinstance(raw_calls, list):
      continue
    call_ids = {
      str(call.get("id") or call.get("tool_call_id"))
      for call in raw_calls
      if isinstance(call, dict) and (call.get("id") or call.get("tool_call_id"))
    }
    while index < len(entries) and entries[index].role == MemoryRole.TOOL:
      tool_call_id = entries[index].metadata.get("tool_call_id")
      if call_ids and str(tool_call_id) not in call_ids:
        break
      index += 1
  return index


def _latest_user_entry(entries: list[MemoryEntry]) -> MemoryEntry | None:
  return next((entry for entry in reversed(entries) if entry.role == MemoryRole.USER), None)


# --------------------------------------------------------------------------- #
# WorkingMemory
# --------------------------------------------------------------------------- #

class WorkingMemory:
  """
  Session-level context window management.

  Responsible for caching, token budget control, and optional interaction
  with a long-term ``MemoryBackend``.

  Budget strategy (sliding-window vs summarisation) is chosen automatically:

  * No ``summary_llm`` → pure sliding-window eviction.
  * With ``summary_llm`` → sliding-window + automatic summary when the
    window grows too large.
  """

  def __init__(
    self,
    session_id: str,
    memory_backend: MemoryBackend | None = None,
    max_tokens: int = 8192,
    summary_llm: "Any | None" = None,
    token_counter: "callable[[str], int] | None" = None,
  ):
    self.session_id = session_id
    self.backend = memory_backend
    self.max_tokens = max_tokens
    self.summary_llm = summary_llm
    self._token_counter = token_counter or _default_count_tokens
    self.entries: list[MemoryEntry] = []

    # Safe background-write bookkeeping: each backend.add() is wrapped in
    # an asyncio task so exceptions are logged (not swallowed) and pending
    # writes can be awaited on shutdown via ``flush()``.
    self._pending_tasks: set[asyncio.Task[None]] = set()

  def _count_tokens(self, content: str) -> int:
    return self._token_counter(content)

  async def _enforce_budget(self) -> None:
    """Evict oldest non-system entries until we are under ``max_tokens``."""
    total = sum(e.tokens for e in self.entries)
    if total <= self.max_tokens:
      return

    system_entries = [e for e in self.entries if e.role == MemoryRole.SYSTEM]
    chat_entries = [e for e in self.entries if e.role != MemoryRole.SYSTEM]
    latest_user = _latest_user_entry(chat_entries)

    # If we have a summary_llm and enough chat history, summarise instead
    # of blindly dropping.
    if self.summary_llm and len(chat_entries) > 2:
      await self._summarise_and_compress(system_entries, chat_entries)
    else:
      while chat_entries and total > self.max_tokens:
        # An assistant tool-call message and every one of its tool responses
        # form one protocol unit.  Evicting only the assistant leaves orphaned
        # ``role=tool`` messages, which OpenAI-compatible APIs reject on the
        # next request.  The ReAct loop defers budget enforcement until a
        # complete batch is present; this fallback also keeps ordinary sliding
        # window eviction structurally valid.
        # Preserve the latest user instruction. It is the task contract for
        # the current turn (often including literal paths or replacement
        # strings), while earlier assistant/tool exchanges are expendable.
        start = 1 if chat_entries[0] is latest_user else 0
        if start >= len(chat_entries):
          break
        removed = _pop_protocol_unit(chat_entries, start)
        total -= sum(entry.tokens for entry in removed)
      self.entries = system_entries + chat_entries

  async def _summarise_and_compress(
    self,
    system_entries: list[MemoryEntry],
    chat_entries: list[MemoryEntry],
  ) -> None:
    """Replace the oldest chunk of chat history with an LLM summary."""
    latest_user = _latest_user_entry(chat_entries)
    protected_prefix = 1 if chat_entries and chat_entries[0] is latest_user else 0
    available = chat_entries[protected_prefix:]
    num_to_summarise = _complete_protocol_prefix_length(
      available, min(5, max(0, len(available) - 1)),
    )
    to_summarise = available[:num_to_summarise]
    kept_chats = chat_entries[:protected_prefix] + available[num_to_summarise:]

    # There is no safe older exchange to summarise without replacing the
    # active user task. Fall back to protocol-aware sliding-window trimming.
    if not to_summarise:
      total = sum(entry.tokens for entry in self.entries)
      while chat_entries and total > self.max_tokens:
        start = 1 if chat_entries[0] is latest_user else 0
        if start >= len(chat_entries):
          break
        removed = _pop_protocol_unit(chat_entries, start)
        total -= sum(entry.tokens for entry in removed)
      self.entries = system_entries + chat_entries
      return

    prompt = (
      "Please summarise the following conversation into a short summary, "
      "preserving core information, entities and conclusions:\n"
      + "\n".join(f"{e.role}: {e.content}" for e in to_summarise)
    )

    from nonoka.core.llm import LLMMessage
    response = await self.summary_llm.chat([LLMMessage(role="user", content=prompt)])

    summary_content = response.content or ""
    summary_entry = MemoryEntry(
      role=MemoryRole.SYSTEM,
      content=f"History Summary: {summary_content}",
      tokens=self._count_tokens(summary_content) if summary_content else 0,
    )

    self.entries = system_entries + [summary_entry] + kept_chats

    # Re-check budget — the summary may still be too long.
    total = sum(e.tokens for e in self.entries)
    if total > self.max_tokens:
      chat_entries_2 = [e for e in self.entries if e.role != MemoryRole.SYSTEM]
      system_entries_2 = [e for e in self.entries if e.role == MemoryRole.SYSTEM]
      while chat_entries_2 and total > self.max_tokens:
        latest_user_2 = _latest_user_entry(chat_entries_2)
        start = 1 if chat_entries_2[0] is latest_user_2 else 0
        if start >= len(chat_entries_2):
          break
        removed = _pop_protocol_unit(chat_entries_2, start)
        total -= sum(entry.tokens for entry in removed)
      self.entries = system_entries_2 + chat_entries_2

  # ------------------------------------------------------------------ #
  # Public API
  # ------------------------------------------------------------------ #

  async def add(
    self,
    content: str,
    role: MemoryRole,
    *,
    defer_budget: bool = False,
    **metadata: Any,
  ) -> None:
    """Add a new message to the context window and (optionally) the backend."""
    tokens = self._count_tokens(content)
    entry = MemoryEntry(role=role, content=content, metadata=metadata, tokens=tokens)
    self.entries.append(entry)
    if not defer_budget:
      await self._enforce_budget()

    # Async push to persistent backend — safe fire-and-forget with
    # exception logging and graceful flush support.
    if self.backend:
      task = asyncio.create_task(
        self._safe_backend_add(content, metadata),
      )
      self._pending_tasks.add(task)
      task.add_done_callback(self._pending_tasks.discard)

  async def enforce_budget(self) -> None:
    """Apply context trimming after callers append an atomic message batch.

    ReAct uses this after all results for one assistant tool-call message have
    been recorded.  It prevents a large early result from evicting the parent
    assistant message before later sibling results are appended.
    """
    await self._enforce_budget()

  async def _safe_backend_add(self, content: str, metadata: dict[str, Any]) -> None:
    """Wrap backend.add() so exceptions are logged, not swallowed."""
    try:
      await self.backend.add(
        content=content,
        session_id=self.session_id,
        metadata=metadata,
      )
    except Exception:
      _logger.exception("memory.backend_write_failed")

  async def flush(self) -> None:
    """Await all pending backend writes. Call before shutdown / checkpoint."""
    if self._pending_tasks:
      await asyncio.gather(*self._pending_tasks, return_exceptions=True)
      self._pending_tasks.clear()

  async def get_context(self) -> list[MemoryEntry]:
    """
    Assemble the full context for the LLM.

    If a backend is configured the latest USER message is used to
    retrieve relevant historical memories and inject them as a system
    prefix.
    """
    if not self.backend:
      return self.entries

    user_msgs = [e for e in self.entries if e.role == MemoryRole.USER]
    if not user_msgs:
      return self.entries

    latest_query = user_msgs[-1].content
    relevant = await self.backend.search(
      query=latest_query,
      session_id=self.session_id,
      limit=3,
    )

    if not relevant:
      return self.entries

    context_str = "\n".join(f"- {m.content}" for m in relevant)
    rag_entry = MemoryEntry(
      role=MemoryRole.SYSTEM,
      content=f"Relevant history memories:\n{context_str}",
      tokens=self._count_tokens(context_str),
    )

    system_entries = [e for e in self.entries if e.role == MemoryRole.SYSTEM]
    chat_entries = [e for e in self.entries if e.role != MemoryRole.SYSTEM]
    return system_entries + [rag_entry] + chat_entries
