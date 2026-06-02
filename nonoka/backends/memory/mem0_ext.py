from typing import Any

from nonoka.core.logger import get_logger
from nonoka.core.memory import MemoryBackend, MemoryEntry, MemoryRole

logger = get_logger(__name__)

try:
  import mem0
  MEM0_AVAILABLE = True
except ImportError:
  MEM0_AVAILABLE = False


class Mem0Backend(MemoryBackend):
  """
  Mem0-based production-ready Memory Backend.
  Support vector retrieval, conflict resolution, automatic summary merging.
  """

  def __init__(self, client: Any = None):
    if not MEM0_AVAILABLE:
      raise ImportError(
        "The mem0ai library is required to use Mem0Backend. "
        "Install it with: pip install mem0ai"
      )

    self._client = client or mem0.MemoryClient()

  async def add(
    self,
    content: str,
    session_id: str | None = None,
    user_id: str | None = None,
    metadata: dict[str, Any] | None = None
  ) -> None:
    """Call mem0 to add memory"""
    merged_metadata = {"session_id": session_id} if session_id else {}
    if metadata:
      merged_metadata.update(metadata)

    import asyncio
    await asyncio.to_thread(
      self._client.add,
      messages=content,
      user_id=user_id,
      metadata=merged_metadata
    )

  async def search(
    self,
    query: str,
    session_id: str | None = None,
    user_id: str | None = None,
    limit: int = 5
  ) -> list[MemoryEntry]:
    """Use Mem0's vector retrieval to search for relevant memories"""
    import asyncio

    results = await asyncio.to_thread(
      self._client.search,
      query=query,
      user_id=user_id,
      limit=limit
    )

    entries = []
    for r in results:
      content = r.get("memory", "")
      meta = r.get("metadata", {})

      if session_id and meta.get("session_id") != session_id:
        continue

      entries.append(
        MemoryEntry(role=MemoryRole.SYSTEM, content=content, metadata=meta)
      )
    return entries

  async def get_history(
    self,
    session_id: str,
    limit: int | None = None
  ) -> list[MemoryEntry]:
    import asyncio

    results = await asyncio.to_thread(
      self._client.get_all,
      {"session_id": session_id}
    )

    entries = [
      MemoryEntry(
        role=MemoryRole.USER,
        content=r.get("memory", ""),
        metadata=r.get("metadata", {})
      )
      for r in results
    ]
    return entries[-limit:] if limit else entries

  async def get_user_memory(
    self,
    user_id: str,
    limit: int = 10
  ) -> list[MemoryEntry]:
    import asyncio

    results = await asyncio.to_thread(
      self._client.get_all,
      user_id=user_id
    )

    entries = [
      MemoryEntry(
        role=MemoryRole.USER,
        content=r.get("memory", ""),
        metadata=r.get("metadata", {})
      )
      for r in results
    ]
    return entries[-limit:] if limit else entries