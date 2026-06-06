"""
Session map for Gateway cross-platform session persistence.

Maps (platform, user_id) tuples to session IDs so that the same user
on the same platform retains context across multiple messages.

Usage::

    sm = SessionMap()
    sm.set("telegram:alice", "sess-123")
    session_id = sm.get("telegram:alice")  # "sess-123"
"""

from __future__ import annotations


class SessionMap:
  """In-memory session ID mapping.

  For production deployments, replace with a persistent store
  (Redis, SQLite, etc.) that survives process restarts.
  """

  def __init__(self):
    self._map: dict[str, str] = {}

  def get(self, key: str) -> str | None:
    """Retrieve session ID for *key*, or ``None`` if not found."""
    return self._map.get(key)

  def set(self, key: str, session_id: str) -> None:
    """Store *session_id* under *key*."""
    self._map[key] = session_id

  def delete(self, key: str) -> None:
    """Remove the mapping for *key*."""
    self._map.pop(key, None)

  def clear(self) -> None:
    """Clear all mappings."""
    self._map.clear()
