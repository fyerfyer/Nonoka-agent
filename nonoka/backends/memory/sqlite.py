from __future__ import annotations

import asyncio
import json
import sqlite3
from typing import Any

from nonoka.core.memory import MemoryBackend, MemoryEntry, MemoryRole


class SQLiteMemoryBackend(MemoryBackend):
  """
  SQLite-backed memory backend.

  Uses Python's built-in ``sqlite3`` module — zero external dependencies.
  Stores memory entries with simple substring search (no vector retrieval).

  Usage::

    # Default: in-memory database
    backend = SQLiteMemoryBackend()

    # File-backed persistence
    backend = SQLiteMemoryBackend(db_path="/path/to/memory.db")

    # Async context manager
    async with SQLiteMemoryBackend("memory.db") as backend:
        await backend.add("User likes blue", session_id="sess-1")
  """

  def __init__(self, db_path: str = ":memory:"):
    self._db_path = db_path
    self._conn: sqlite3.Connection | None = None
    self._lock = asyncio.Lock()

  # ------------------------------------------------------------------ #
  # Connection management
  # ------------------------------------------------------------------ #

  def _ensure_connection(self) -> sqlite3.Connection:
    """Return an open connection, creating one if necessary."""
    if self._conn is None:
      self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
      self._conn.row_factory = sqlite3.Row
      self._create_tables()
    return self._conn

  def _create_tables(self) -> None:
    """Create required tables if they do not exist."""
    conn = self._conn
    assert conn is not None
    conn.execute(
      """
      CREATE TABLE IF NOT EXISTS memory_entries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        content TEXT NOT NULL,
        session_id TEXT,
        user_id TEXT,
        metadata_json TEXT DEFAULT '{}',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      )
      """
    )
    conn.execute(
      """
      CREATE INDEX IF NOT EXISTS idx_memory_session_id
      ON memory_entries(session_id)
      """
    )
    conn.execute(
      """
      CREATE INDEX IF NOT EXISTS idx_memory_user_id
      ON memory_entries(user_id)
      """
    )
    conn.commit()

  async def close(self) -> None:
    """Close the database connection."""
    if self._conn is not None:
      await asyncio.to_thread(self._conn.close)
      self._conn = None

  async def __aenter__(self) -> SQLiteMemoryBackend:
    return self

  async def __aexit__(self, *args: Any) -> None:
    await self.close()

  # ------------------------------------------------------------------ #
  # Helpers
  # ------------------------------------------------------------------ #

  @staticmethod
  def _row_to_entry(row: sqlite3.Row) -> MemoryEntry:
    metadata = json.loads(row["metadata_json"] or "{}")
    return MemoryEntry(
      role=MemoryRole.USER,
      content=row["content"],
      metadata=metadata,
    )

  # ------------------------------------------------------------------ #
  # Protocol implementation
  # ------------------------------------------------------------------ #

  async def add(
    self,
    content: str,
    session_id: str | None = None,
    user_id: str | None = None,
    metadata: dict[str, Any] | None = None,
  ) -> None:
    """Add a memory entry."""

    def _add() -> None:
      conn = self._ensure_connection()
      conn.execute(
        """
        INSERT INTO memory_entries (content, session_id, user_id, metadata_json)
        VALUES (?, ?, ?, ?)
        """,
        (content, session_id, user_id, json.dumps(metadata or {})),
      )
      conn.commit()

    async with self._lock:
      await asyncio.to_thread(_add)

  async def search(
    self,
    query: str,
    session_id: str | None = None,
    user_id: str | None = None,
    limit: int = 5,
  ) -> list[MemoryEntry]:
    """Search memory entries by substring matching."""

    def _search() -> list[MemoryEntry]:
      conn = self._ensure_connection()
      # Build query dynamically based on filters
      conditions: list[str] = ["content LIKE ?"]
      params: list[Any] = [f"%{query}%"]

      if session_id is not None:
        conditions.append("session_id = ?")
        params.append(session_id)
      if user_id is not None:
        conditions.append("user_id = ?")
        params.append(user_id)

      where_clause = " AND ".join(conditions)
      params.append(limit)

      rows = conn.execute(
        f"""
        SELECT content, metadata_json
        FROM memory_entries
        WHERE {where_clause}
        ORDER BY id DESC
        LIMIT ?
        """,
        tuple(params),
      ).fetchall()

      return [self._row_to_entry(row) for row in rows]

    return await asyncio.to_thread(_search)

  async def get_history(
    self,
    session_id: str,
    limit: int | None = None,
  ) -> list[MemoryEntry]:
    """Get all memory entries for a session, oldest first.

    When *limit* is given, returns the most recent *limit* entries
    but still ordered oldest → newest.
    """

    def _get_history() -> list[MemoryEntry]:
      conn = self._ensure_connection()
      if limit and limit > 0:
        # Subquery: grab the most recent N by id, then re-order ASC
        rows = conn.execute(
          """
          SELECT content, metadata_json FROM (
            SELECT content, metadata_json, id
            FROM memory_entries
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
          ) sub
          ORDER BY sub.id ASC
          """,
          (session_id, limit),
        ).fetchall()
      else:
        rows = conn.execute(
          """
          SELECT content, metadata_json
          FROM memory_entries
          WHERE session_id = ?
          ORDER BY id ASC
          """,
          (session_id,),
        ).fetchall()

      return [self._row_to_entry(row) for row in rows]

    return await asyncio.to_thread(_get_history)

  async def get_user_memory(
    self,
    user_id: str,
    limit: int = 10,
  ) -> list[MemoryEntry]:
    """Get all memory entries for a user, most recent first."""

    def _get_user_memory() -> list[MemoryEntry]:
      conn = self._ensure_connection()
      rows = conn.execute(
        """
        SELECT content, metadata_json
        FROM memory_entries
        WHERE user_id = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (user_id, limit),
      ).fetchall()

      return [self._row_to_entry(row) for row in rows]

    return await asyncio.to_thread(_get_user_memory)
