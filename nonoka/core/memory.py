from enum import Enum
from typing import Any, Protocol, runtime_checkable, TYPE_CHECKING
from pydantic import BaseModel, Field

if TYPE_CHECKING:
  from nonoka.core.llm import LLMProvider, LLMMessage


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
  """Persistent memory storage interface"""
  async def add(self, content: str, session_id: str | None = None, user_id: str | None = None, metadata: dict[str, Any] | None = None) -> None: ...
  async def search(self, query: str, session_id: str | None = None, user_id: str | None = None, limit: int = 5) -> list[MemoryEntry]: ...
  async def get_history(self, session_id: str, limit: int | None = None) -> list[MemoryEntry]: ...
  async def get_user_memory(self, user_id: str, limit: int = 10) -> list[MemoryEntry]: ...

class MemoryBudgetStrategy(Protocol):
  """Token budget management strategy protocol."""
  async def enforce(self, memory: "WorkingMemory") -> None: ...


class DefaultBudgetStrategy(MemoryBudgetStrategy):
  """Default budget strategy: use sliding window to remove the oldest messages"""
  def __init__(self, max_tokens: int):
    self.max_tokens = max_tokens

  async def enforce(self, memory: "WorkingMemory") -> None:
    """Enforce token budget by removing the oldest messages."""
    total_token = sum(e.tokens for e in memory.entries)

    if total_token > self.max_tokens:
      system_entries = [e for e in memory.entries if e.role == MemoryRole.SYSTEM]
      chat_entries = [e for e in memory.entries if e.role != MemoryRole.SYSTEM]

      while chat_entries and total_token > self.max_tokens:
        removed = chat_entries.pop(0)
        total_token -= removed.tokens

      memory.entries = system_entries + chat_entries
    
       
class SummaryBudgetStrategy(MemoryBudgetStrategy):
  """LLM-based conversation summary compression strategy"""
  def __init__(self, max_tokens: int, summary_llm: "LLMProvider"):
    self.max_tokens = max_tokens
    self.summary_llm = summary_llm

  async def enforce(self, memory: "WorkingMemory") -> None:
    total_token = sum(e.tokens for e in memory.entries)
    
    if total_token <= self.max_tokens:
      return

    system_entries = [e for e in memory.entries if e.role == MemoryRole.SYSTEM]
    chat_entries = [e for e in memory.entries if e.role != MemoryRole.SYSTEM]

    if len(chat_entries) <= 2:
      # Not enough chats to summarize meaningfully, fallback to sliding window
      while chat_entries and total_token > self.max_tokens:
        removed = chat_entries.pop(0)
        total_token -= removed.tokens
      memory.entries = system_entries + chat_entries
      return

    # Take the oldest 5 turns to summarize
    num_to_summarize = min(5, len(chat_entries) - 1) 
    to_summarize = chat_entries[:num_to_summarize]
    kept_chats = chat_entries[num_to_summarize:]
    
    from nonoka.core.llm import LLMMessage
    prompt = "Please summarize the following conversation into a short summary, preserving core information, entities and conclusions:\n" + "\n".join([f"{e.role}: {e.content}" for e in to_summarize])
    
    response = await self.summary_llm.chat([
        LLMMessage(role="user", content=prompt)
    ])
    
    summary_content = response.content or ""
    summary_entry = MemoryEntry(
        role=MemoryRole.SYSTEM, 
        content=f"History Summary: {summary_content}",
        tokens=self.summary_llm.count_tokens(summary_content) if summary_content else 0
    )
    
    memory.entries = system_entries + [summary_entry] + kept_chats

class WorkingMemory:
  """
  Session-level context window management (Working Memory).
  Responsible for caching, token control, and interaction with long-term memory.
  """
  def __init__(
    self,
    session_id: str,
    memory_backend: MemoryBackend | None = None,
    max_tokens: int = 8192,
    budget_strategy: MemoryBudgetStrategy | None = None, 
    summary_llm: "LLMProvider | None" = None,
  ):
    self.session_id = session_id
    self.backend = memory_backend
    self.max_tokens = max_tokens
    self.summary_llm = summary_llm
    self.entries: list[MemoryEntry] = []

    # Smart routing
    if budget_strategy:
      self.strategy = budget_strategy
    elif self.summary_llm:
      self.strategy = SummaryBudgetStrategy(max_tokens=max_tokens, summary_llm=self.summary_llm)
    else:
      self.strategy = DefaultBudgetStrategy(max_tokens=max_tokens)

  def _count_tokens(self, content: str) -> int:
    """
    Calculate the number of tokens in the text.
    """
    if self.summary_llm:
      return self.summary_llm.count_tokens(content)
    return len(content) // 3 if content else 0

  async def add(self, content: str, role: MemoryRole, **metadata: Any) -> None:
    """Add new message to context and synchronize to long-term memory asynchronously"""

    # Check and release token space
    tokens = self._count_tokens(content)
    entry = MemoryEntry(role=role, content=content, metadata=metadata, tokens=tokens)
    self.entries.append(entry)
    await self.strategy.enforce(self)

    # Asynchronously push to persistent backend
    if self.backend:
      import asyncio
      asyncio.create_task(
        self.backend.add(
          content=content,
          session_id=self.session_id,
          metadata=metadata
        )
      )

  async def get_context(self) -> list[MemoryEntry]:
    """
    Intelligently assemble the context for LLM.
    Strategy:
    1. Always keep SYSTEM prompt.
    2. Keep recent conversations.
    3. Use the latest USER message to retrieve relevant history from MemoryBackend as a supplement.
    """
    if not self.backend:
      return self.entries

    # Get the latest USER message
    user_msgs = [e for e in self.entries if e.role == MemoryRole.USER]
    if not user_msgs:
      return self.entries

    latest_query = user_msgs[-1].content

    # Retrieve relevant history from long-term memory 
    relevant_memories = await self.backend.search(
      query=latest_query,
      session_id=self.session_id,
      limit=3
    )

    if not relevant_memories:
      return self.entries

    # If additional information is retrieved, we can insert it into the current context as a System prompt or special format
    # Simple strategy: combine retrieved memories into a System prompt
    context_str = "\n".join([f"- {m.content}" for m in relevant_memories])
    rag_entry = MemoryEntry(
      role=MemoryRole.SYSTEM,
      content=f"Relevant history memories:\n{context_str}",
      tokens=self._count_tokens(context_str)
    )

    # Assemble: [Original System] + [Retrieved Memories] + [Chat History]
    system_entries = [e for e in self.entries if e.role == MemoryRole.SYSTEM]
    chat_entries = [e for e in self.entries if e.role != MemoryRole.SYSTEM]

    return system_entries + [rag_entry] + chat_entries

  def _enforce_budget(self) -> None:
    """
    Token budget management.
    If current token count exceeds max_tokens, the oldest conversation (except SYSTEM Prompt) will be eliminated.
    """
    total_tokens = sum(e.tokens for e in self.entries)
    if total_tokens <= self.max_tokens:
      return

    system_entries = [e for e in self.entries if e.role == MemoryRole.SYSTEM]
    chat_entries = [e for e in self.entries if e.role != MemoryRole.SYSTEM]

    # Sliding window mechanism: keep removing the oldest conversation from the beginning until the budget is met
    while chat_entries and total_tokens > self.max_tokens:
      removed = chat_entries.pop(0)
      total_tokens -= removed.tokens

    self.entries = system_entries + chat_entries