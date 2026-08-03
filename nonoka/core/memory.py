from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Protocol, runtime_checkable
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


@dataclass(frozen=True)
class ContextBudget:
  max_bytes: int | None = None
  max_tokens: int | None = None
  max_tool_messages: int | None = None


@dataclass(frozen=True)
class ContextMetrics:
  serialized_bytes: int
  tokens: int
  tool_messages: int


@dataclass
class ContextCompactionResult:
  entries: list[MemoryEntry]
  metrics: ContextMetrics
  compacted_entries: int = 0
  exceeded: bool = False


@runtime_checkable
class ContextCompactor(Protocol):
  """Replace history with a provider-valid representation within a budget."""

  async def compact(
    self,
    entries: list[MemoryEntry],
    budget: ContextBudget,
    count_tokens: Callable[[str], int],
  ) -> ContextCompactionResult: ...


class ProtocolAwareContextCompactor:
  """Deterministic compactor preserving task and tool protocol integrity."""

  _LEDGER_MAX_BYTES = 12 * 1024

  @staticmethod
  def _metrics(entries: list[MemoryEntry]) -> ContextMetrics:
    payload = [entry.model_dump(mode="json") for entry in entries]
    size = len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    return ContextMetrics(
      serialized_bytes=size,
      tokens=sum(entry.tokens for entry in entries),
      tool_messages=sum(entry.role == MemoryRole.TOOL for entry in entries),
    )

  @staticmethod
  def _within(metrics: ContextMetrics, budget: ContextBudget) -> bool:
    return (
      (budget.max_bytes is None or metrics.serialized_bytes <= budget.max_bytes)
      and (budget.max_tokens is None or metrics.tokens <= budget.max_tokens)
      and (budget.max_tool_messages is None or metrics.tool_messages <= budget.max_tool_messages)
    )

  @staticmethod
  def _tool_calls(entry: MemoryEntry) -> list[dict[str, Any]]:
    calls = entry.metadata.get("tool_calls")
    return [call for call in calls if isinstance(call, dict)] if isinstance(calls, list) else []

  @classmethod
  def _is_complete_unit(cls, entries: list[MemoryEntry], start: int) -> bool:
    entry = entries[start]
    calls = cls._tool_calls(entry)
    if entry.role != MemoryRole.ASSISTANT or not calls:
      return True
    expected = {
      str(call.get("id") or call.get("tool_call_id"))
      for call in calls if call.get("id") or call.get("tool_call_id")
    }
    actual: set[str] = set()
    index = start + 1
    while index < len(entries) and entries[index].role == MemoryRole.TOOL:
      tool_call_id = entries[index].metadata.get("tool_call_id")
      if tool_call_id:
        actual.add(str(tool_call_id))
      index += 1
    return not expected or expected.issubset(actual)

  @classmethod
  def _unit_end(cls, entries: list[MemoryEntry], start: int) -> int:
    """Return the exclusive end of one provider protocol unit."""
    end = start + 1
    entry = entries[start]
    if entry.role != MemoryRole.ASSISTANT or not cls._tool_calls(entry):
      return end
    while end < len(entries) and entries[end].role == MemoryRole.TOOL:
      end += 1
    return end

  @classmethod
  def _is_protected_unit(cls, entries: list[MemoryEntry], start: int) -> bool:
    return any(
      entry.metadata.get("context_protected")
      for entry in entries[start:cls._unit_end(entries, start)]
    )

  @staticmethod
  def _preview(content: str) -> str:
    if len(content) <= 800:
      return content
    return f"{content[:360]}\n...[omitted]...\n{content[-360:]}"

  @classmethod
  def _ledger_items(cls, removed: list[MemoryEntry]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    call_args: dict[str, dict[str, Any]] = {}
    for entry in removed:
      if entry.role == MemoryRole.ASSISTANT:
        for call in cls._tool_calls(entry):
          call_id = str(call.get("id") or call.get("tool_call_id") or "")
          function = call.get("function") if isinstance(call.get("function"), dict) else {}
          args = function.get("arguments")
          if isinstance(args, str):
            try:
              args = json.loads(args)
            except json.JSONDecodeError:
              args = {"raw": args[:500]}
          call_args[call_id] = {
            "tool": function.get("name"),
            "arguments": args if isinstance(args, dict) else {},
          }
      elif entry.role == MemoryRole.TOOL:
        call_id = str(entry.metadata.get("tool_call_id") or "")
        item = dict(call_args.get(call_id, {}))
        item.update({
          "tool_call_id": call_id or None,
          "artifact_ref": entry.metadata.get("artifact_ref"),
          "exit_code": entry.metadata.get("exit_code"),
          "workspace_changes": entry.metadata.get("workspace_changes"),
          "result": cls._preview(entry.content),
        })
        items.append(item)
      elif entry.role in {MemoryRole.USER, MemoryRole.ASSISTANT} and entry.content:
        items.append({"role": entry.role.value, "content": cls._preview(entry.content)})
    return items

  async def compact(
    self,
    entries: list[MemoryEntry],
    budget: ContextBudget,
    count_tokens: Callable[[str], int],
  ) -> ContextCompactionResult:
    current = list(entries)
    metrics = self._metrics(current)
    if self._within(metrics, budget):
      return ContextCompactionResult(current, metrics)

    previous_ledgers = [
      entry for entry in current if entry.metadata.get("evidence_ledger")
    ]
    current = [entry for entry in current if not entry.metadata.get("evidence_ledger")]
    system_entries = [entry for entry in current if entry.role == MemoryRole.SYSTEM]
    chat_entries = [entry for entry in current if entry.role != MemoryRole.SYSTEM]
    latest_user = _latest_user_entry(chat_entries)
    removed: list[MemoryEntry] = []

    while chat_entries and not self._within(self._metrics(system_entries + chat_entries), budget):
      start = 0
      while start < len(chat_entries):
        if chat_entries[start] is latest_user:
          start = self._unit_end(chat_entries, start)
          continue
        if not self._is_complete_unit(chat_entries, start):
          start = self._unit_end(chat_entries, start)
          continue
        if self._is_protected_unit(chat_entries, start):
          start = self._unit_end(chat_entries, start)
          continue
        break
      if start >= len(chat_entries):
        break
      removed.extend(_pop_protocol_unit(chat_entries, start))

    ledger_items: list[dict[str, Any]] = []
    for entry in previous_ledgers:
      _, _, payload = entry.content.partition("\n")
      try:
        prior = json.loads(payload)
      except json.JSONDecodeError:
        prior = []
      if isinstance(prior, list):
        ledger_items.extend(item for item in prior if isinstance(item, dict))
    ledger_items.extend(self._ledger_items(removed))

    if ledger_items:
      while ledger_items:
        ledger_text = "[Compacted evidence ledger]\n" + json.dumps(
          ledger_items, ensure_ascii=False, separators=(",", ":"), default=str,
        )
        if len(ledger_text.encode("utf-8")) <= self._LEDGER_MAX_BYTES:
          break
        ledger_items.pop(0)
      if ledger_items:
        ledger = MemoryEntry(
          role=MemoryRole.SYSTEM,
          content=ledger_text,
          tokens=count_tokens(ledger_text),
          metadata={"evidence_ledger": True, "compacted_entries": len(removed)},
        )
        system_entries.append(ledger)

    compacted = system_entries + chat_entries
    metrics = self._metrics(compacted)
    if not self._within(metrics, budget):
      compacted = [entry for entry in compacted if not entry.metadata.get("evidence_ledger")]
      metrics = self._metrics(compacted)
    return ContextCompactionResult(
      compacted,
      metrics,
      compacted_entries=len(removed),
      exceeded=not self._within(metrics, budget),
    )


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
# Microcompaction — superseded tool results
# --------------------------------------------------------------------------- #

_READ_TOOLS = {"read", "read_file", "view"}
_SEARCH_TOOLS = {"grep", "glob", "grep_files", "search_files"}
_SHELL_TOOLS = {"bash", "shell", "execute_command", "run_command"}


def _tool_call_arguments(entries: list[MemoryEntry]) -> dict[str, tuple[str, dict[str, Any]]]:
  """Map ``tool_call_id`` to ``(tool name, parsed arguments)`` from assistants."""
  calls: dict[str, tuple[str, dict[str, Any]]] = {}
  for entry in entries:
    if entry.role != MemoryRole.ASSISTANT:
      continue
    for call in ProtocolAwareContextCompactor._tool_calls(entry):
      call_id = call.get("id") or call.get("tool_call_id")
      function = call.get("function") if isinstance(call.get("function"), dict) else {}
      name = function.get("name")
      arguments = function.get("arguments")
      if isinstance(arguments, str):
        try:
          arguments = json.loads(arguments)
        except json.JSONDecodeError:
          arguments = None
      if call_id and name:
        calls[str(call_id)] = (str(name), arguments if isinstance(arguments, dict) else {})
  return calls


def _superseded_key(tool_name: str, arguments: dict[str, Any]) -> tuple[str, ...] | None:
  """Logical dedup key for one tool result, or ``None`` when unreliable."""
  name = tool_name.lower()
  if name in _READ_TOOLS:
    path = arguments.get("file_path") or arguments.get("path")
    return (name, str(path)) if path else None
  if name in _SEARCH_TOOLS:
    pattern = arguments.get("pattern")
    if not pattern:
      return None
    return (name, str(pattern), str(arguments.get("path") or ""))
  if name in _SHELL_TOOLS:
    command = arguments.get("command")
    return (name, str(command)[:200]) if command else None
  return None


def microcompact_superseded_tool_results(
  entries: list[MemoryEntry],
  count_tokens: Callable[[str], int] = _default_count_tokens,
) -> list[MemoryEntry]:
  """Replace superseded duplicate tool results with a short placeholder.

  Repeating the same logical tool call (e.g. Read on the same file) makes
  every result but the newest stale.  Older contents are replaced in place —
  entries are never removed — so assistant/tool protocol units stay intact.
  Results without reliable call arguments are left untouched.
  """
  call_args = _tool_call_arguments(entries)
  keys: list[tuple[str, ...] | None] = []
  for entry in entries:
    key: tuple[str, ...] | None = None
    if entry.role == MemoryRole.TOOL and not entry.metadata.get("superseded"):
      call_id = str(entry.metadata.get("tool_call_id") or "")
      name, arguments = call_args.get(
        call_id, (entry.metadata.get("tool_name"), {}),
      )
      if name:
        key = _superseded_key(str(name), arguments)
    keys.append(key)

  latest: dict[tuple[str, ...], int] = {}
  for index, key in enumerate(keys):
    if key is not None:
      latest[key] = index

  result = list(entries)
  for index, key in enumerate(keys):
    if key is None or latest[key] == index:
      continue
    entry = result[index]
    placeholder = f"[superseded by newer {key[0]} result]"
    metadata = dict(entry.metadata)
    metadata["superseded"] = True
    result[index] = entry.model_copy(update={
      "content": placeholder,
      "tokens": count_tokens(placeholder),
      "metadata": metadata,
    })
  return result


# --------------------------------------------------------------------------- #
# WorkingMemory
# --------------------------------------------------------------------------- #

class WorkingMemory:
  """
  Session-level context window management.

  Responsible for caching, token budget control, and optional interaction
  with a long-term ``MemoryBackend``.

  Budget strategy (sliding-window vs summarisation) is chosen automatically:

  * No ``summary_llm`` → deterministic protocol-aware compaction.
  * With ``summary_llm`` → automatic summary of the oldest history when the
    window grows too large (falls back to deterministic compaction after
    repeated summariser failures).

  Every budget check first microcompacts superseded duplicate tool results,
  triggers ``compaction_buffer_tokens`` before the cap, and compacts down to
  ``max_tokens - reserve_output_tokens``.
  """

  def __init__(
    self,
    session_id: str,
    memory_backend: MemoryBackend | None = None,
    max_tokens: int = 8192,
    summary_llm: "Any | None" = None,
    token_counter: "callable[[str], int] | None" = None,
    context_compactor: ContextCompactor | None = None,
    reserve_output_tokens: int = 4096,
    compaction_buffer_tokens: int = 2048,
  ):
    self.session_id = session_id
    self.backend = memory_backend
    self.max_tokens = max_tokens
    self.summary_llm = summary_llm
    self._token_counter = token_counter or _default_count_tokens
    self.context_compactor = context_compactor or ProtocolAwareContextCompactor()
    self.reserve_output_tokens = reserve_output_tokens
    self.compaction_buffer_tokens = compaction_buffer_tokens
    self.entries: list[MemoryEntry] = []

    # Circuit breaker: after three consecutive summary failures this session
    # stops calling the summariser and uses the deterministic compactor.
    self._summary_failures = 0
    self._summary_disabled = False

    # Safe background-write bookkeeping: each backend.add() is wrapped in
    # an asyncio task so exceptions are logged (not swallowed) and pending
    # writes can be awaited on shutdown via ``flush()``.
    self._pending_tasks: set[asyncio.Task[None]] = set()

  def _count_tokens(self, content: str) -> int:
    return self._token_counter(content)

  def _compaction_target(self) -> int:
    """Token target for one compaction pass.

    A tiny ``max_tokens`` cannot hold the reserve; clamp to the raw cap so
    the budget stays reachable instead of aiming at a negative target.
    """
    target = self.max_tokens - self.reserve_output_tokens
    return target if target > 0 else self.max_tokens

  async def _enforce_budget(self) -> None:
    """Compact the window once it comes within ``compaction_buffer_tokens``
    of ``max_tokens``, trimming down to ``max_tokens - reserve_output_tokens``
    so the next model response has room.
    """
    # Microcompaction first: superseded duplicate tool results are cheap
    # wins that often make real compaction unnecessary.
    self.entries = microcompact_superseded_tool_results(self.entries, self._count_tokens)
    total = sum(e.tokens for e in self.entries)
    target = self._compaction_target()
    # Clamp the trigger at the target so a buffer larger than the reserve
    # cannot cause repeated no-progress compaction passes.
    if total <= max(self.max_tokens - self.compaction_buffer_tokens, target):
      return

    if self.summary_llm is None or self._summary_disabled:
      result = await self.context_compactor.compact(
        list(self.entries), ContextBudget(max_tokens=target), self._count_tokens,
      )
      self.entries = result.entries
      return

    system_entries = [e for e in self.entries if e.role == MemoryRole.SYSTEM]
    chat_entries = [e for e in self.entries if e.role != MemoryRole.SYSTEM]
    latest_user = _latest_user_entry(chat_entries)

    # If we have a summary_llm and enough chat history, summarise instead
    # of blindly dropping.
    if len(chat_entries) > 2:
      await self._summarise_and_compress(system_entries, chat_entries, target)
    else:
      while chat_entries and total > target:
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
    target: int,
  ) -> None:
    """Replace the oldest chunk of chat history with an LLM summary."""
    latest_user = _latest_user_entry(chat_entries)
    protected_prefix = 1 if chat_entries and chat_entries[0] is latest_user else 0
    available = chat_entries[protected_prefix:]
    # Summarise roughly the oldest third of the chat history, rounded up to
    # complete assistant/tool protocol units.
    num_to_summarise = _complete_protocol_prefix_length(
      available, max(0, len(available) // 3),
    )
    to_summarise = available[:num_to_summarise]
    kept_chats = chat_entries[:protected_prefix] + available[num_to_summarise:]

    # There is no safe older exchange to summarise without replacing the
    # active user task. Fall back to protocol-aware sliding-window trimming.
    if not to_summarise:
      total = sum(entry.tokens for entry in self.entries)
      while chat_entries and total > target:
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
    try:
      response = await self.summary_llm.chat([LLMMessage(role="user", content=prompt)])
    except Exception:
      # A broken summariser must not break the session: count the failure
      # towards the circuit breaker and fall back to deterministic
      # ledger compaction.
      self._summary_failures += 1
      if self._summary_failures >= 3:
        self._summary_disabled = True
      _logger.exception("memory.summary_failed")
      result = await self.context_compactor.compact(
        system_entries + chat_entries,
        ContextBudget(max_tokens=target),
        self._count_tokens,
      )
      self.entries = result.entries
      return
    self._summary_failures = 0

    summary_content = response.content or ""
    summary_entry = MemoryEntry(
      role=MemoryRole.SYSTEM,
      content=f"History Summary: {summary_content}",
      tokens=self._count_tokens(summary_content) if summary_content else 0,
    )

    self.entries = system_entries + [summary_entry] + kept_chats

    # Re-check budget — the summary may still be too long.
    total = sum(e.tokens for e in self.entries)
    if total > target:
      chat_entries_2 = [e for e in self.entries if e.role != MemoryRole.SYSTEM]
      system_entries_2 = [e for e in self.entries if e.role == MemoryRole.SYSTEM]
      while chat_entries_2 and total > target:
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

  async def enforce_budget(self, runtime_limits: Any | None = None) -> ContextMetrics | None:
    """Apply context trimming after callers append an atomic message batch.

    ReAct uses this after all results for one assistant tool-call message have
    been recorded.  It prevents a large early result from evicting the parent
    assistant message before later sibling results are appended.
    """
    if runtime_limits is None:
      await self._enforce_budget()
      return None

    base_max_tokens = getattr(runtime_limits, "max_context_tokens", None) or self.max_tokens
    # Reserve room for the next model response, mirroring _enforce_budget.
    target = base_max_tokens - self.reserve_output_tokens
    budget = ContextBudget(
      max_bytes=getattr(runtime_limits, "max_context_bytes", None),
      max_tokens=target if target > 0 else base_max_tokens,
      max_tool_messages=getattr(runtime_limits, "max_tool_messages", None),
    )
    result = await self.context_compactor.compact(
      list(self.entries), budget, self._count_tokens,
    )
    self.entries = result.entries
    if result.exceeded:
      from nonoka.core.errors import ContextBudgetExceeded
      raise ContextBudgetExceeded(
        metrics={
          "bytes": result.metrics.serialized_bytes,
          "tokens": result.metrics.tokens,
          "tool_messages": result.metrics.tool_messages,
        },
        limits={
          "max_bytes": budget.max_bytes,
          "max_tokens": budget.max_tokens,
          "max_tool_messages": budget.max_tool_messages,
        },
      )
    return result.metrics

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
