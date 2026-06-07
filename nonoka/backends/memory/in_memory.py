from __future__ import annotations

from typing import Any
from nonoka.core.memory import MemoryBackend, MemoryEntry, MemoryRole


class InMemoryBackend(MemoryBackend):
  """
  Light-weight Memory Backend, stored in memory
  Do not support vector retrieval, using simple string matching
  """

  def __init__(self):
    self._entries: list[dict[str, Any]] = []

  async def add(
    self,
    content: str,
    session_id: str | None = None,
    user_id: str | None = None,
    metadata: dict[str, Any] | None = None
  ) -> None:
    self._entries.append({
      "content": content,
      "session_id": session_id,
      "user_id": user_id,
      "metadata": metadata or {}
    })

  async def search(
    self,
    query: str,
    session_id: str | None = None,
    user_id: str | None = None,
    limit: int = 5
  ) -> list[MemoryEntry]:
    """String matching fallback"""
    results = []
    for e in reversed(self._entries):
      if session_id and e["session_id"] != session_id:
        continue
      if user_id and e["user_id"] != user_id:
        continue

      if query.lower() in e["content"].lower():
        results.append(
          MemoryEntry(role=MemoryRole.USER, content=e["content"], metadata=e["metadata"])
        )
        if len(results) >= limit:
          break

    return results

  async def get_history(
    self,
    session_id: str,
    limit: int | None = None
  ) -> list[MemoryEntry]:
    results = [
      MemoryEntry(role=MemoryRole.USER, content=e["content"], metadata=e["metadata"])
      for e in self._entries if e["session_id"] == session_id
    ]
    return results[-limit:] if limit and limit > 0 else results

  async def get_user_memory(
    self,
    user_id: str,
    limit: int = 10
  ) -> list[MemoryEntry]:
    results = [
      MemoryEntry(role=MemoryRole.USER, content=e["content"], metadata=e["metadata"])
      for e in self._entries if e["user_id"] == user_id
    ]
    return results[-limit:] if limit and limit > 0 else results