from __future__ import annotations

import asyncio
import json
import sqlite3
import traceback as _traceback
from typing import Any

from nonoka.core.checkpoint import CheckpointStore
from nonoka.core.session import SessionState, StepStatus, StepResult, StepFailure


class SQLiteCheckpointStore(CheckpointStore):
  """
  SQLite-backed checkpoint store.

  Uses Python's built-in ``sqlite3`` module — zero external dependencies.
  All sync database operations are executed via ``asyncio.to_thread`` so
  the class presents an async interface compatible with the Protocol.

  Usage::

    # Default: in-memory database (useful for testing)
    store = SQLiteCheckpointStore()

    # File-backed persistence
    store = SQLiteCheckpointStore(db_path="/path/to/checkpoints.db")

    # Async context manager (auto-closes connection)
    async with SQLiteCheckpointStore("checkpoints.db") as store:
        await store.save_session("sess-1", state)
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
      CREATE TABLE IF NOT EXISTS checkpoints (
        session_id TEXT PRIMARY KEY,
        state_json TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      )
      """
    )
    conn.execute(
      """
      CREATE TABLE IF NOT EXISTS step_updates (
        session_id TEXT NOT NULL,
        step_id TEXT NOT NULL,
        update_type TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (session_id, step_id, update_type)
      )
      """
    )
    conn.commit()

  async def close(self) -> None:
    """Close the database connection."""
    if self._conn is not None:
      await asyncio.to_thread(self._conn.close)
      self._conn = None

  async def __aenter__(self) -> SQLiteCheckpointStore:
    return self

  async def __aexit__(self, *args: Any) -> None:
    await self.close()

  # ------------------------------------------------------------------ #
  # Helpers
  # ------------------------------------------------------------------ #

  @staticmethod
  def _state_to_json(state: SessionState) -> str:
    def _fallback(obj: Any) -> Any:
      from nonoka.core.plan import Ref
      if isinstance(obj, Ref):
        return {"__type__": "ref", "step_id": obj.step_id, "path": obj.path}
      raise TypeError(f"Cannot serialize {type(obj).__name__}")

    return state.model_dump_json(fallback=_fallback)

  @staticmethod
  def _state_from_json(data: str) -> SessionState:
    import json
    from nonoka.core.plan import Ref

    raw = json.loads(data)

    def _restore_refs(obj: Any) -> Any:
      if isinstance(obj, dict) and obj.get("__type__") == "ref":
        return Ref(step_id=obj["step_id"], path=obj.get("path", ""))
      if isinstance(obj, dict):
        return {k: _restore_refs(v) for k, v in obj.items()}
      if isinstance(obj, list):
        return [_restore_refs(item) for item in obj]
      return obj

    restored = _restore_refs(raw)
    return SessionState.model_validate(restored)

  # ------------------------------------------------------------------ #
  # Protocol implementation
  # ------------------------------------------------------------------ #

  async def delete_session(self, session_id: str) -> bool:
    """Delete all persisted data for a session.

    Returns:
      True if any rows were deleted, False if the session did not exist.
    """

    def _delete() -> bool:
      conn = self._ensure_connection()
      cursor = conn.execute(
        "DELETE FROM checkpoints WHERE session_id = ?",
        (session_id,),
      )
      conn.execute(
        "DELETE FROM step_updates WHERE session_id = ?",
        (session_id,),
      )
      conn.commit()
      return cursor.rowcount > 0

    async with self._lock:
      return await asyncio.to_thread(_delete)

  async def save_session(self, session_id: str, state: SessionState) -> None:
    """Persist a full session snapshot."""

    def _save() -> None:
      conn = self._ensure_connection()
      conn.execute(
        """
        INSERT INTO checkpoints (session_id, state_json)
        VALUES (?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
          state_json = excluded.state_json,
          updated_at = CURRENT_TIMESTAMP
        """,
        (session_id, self._state_to_json(state)),
      )
      conn.commit()

    async with self._lock:
      await asyncio.to_thread(_save)

  async def load_session(self, session_id: str) -> SessionState | None:
    """Restore a session snapshot, merging any granular step updates."""

    def _load() -> SessionState | None:
      conn = self._ensure_connection()
      row = conn.execute(
        "SELECT state_json FROM checkpoints WHERE session_id = ?",
        (session_id,),
      ).fetchone()

      if row is None:
        return None

      state = self._state_from_json(row["state_json"])

      # Merge step-level updates (avoids N+1 load-mutate-save problem).
      # ORDER BY updated_at ASC so later updates overwrite earlier ones.
      step_rows = conn.execute(
        "SELECT step_id, update_type, payload_json FROM step_updates WHERE session_id = ? ORDER BY updated_at ASC",
        (session_id,),
      ).fetchall()

      for step_row in step_rows:
        step_id = step_row["step_id"]
        update_type = step_row["update_type"]
        payload = json.loads(step_row["payload_json"])

        if update_type == "status":
          new_status = StepStatus(payload["status"])
          # Don't overwrite a terminal status (COMPLETED / FAILED) with a
          # non-terminal one.  A later "status" row may race with an earlier
          # "result" / "error" row when timestamps have identical resolution.
          current = state.step_statuses.get(step_id)
          if current in (StepStatus.COMPLETED, StepStatus.FAILED):
            continue
          state.step_statuses[step_id] = new_status
        elif update_type == "result":
          state.step_statuses[step_id] = StepStatus.COMPLETED
          state.completed_steps[step_id] = StepResult(data=payload["result"])
          # A successful retry should clear any previous failure record.
          state.failed_steps.pop(step_id, None)
        elif update_type == "error":
          state.step_statuses[step_id] = StepStatus.FAILED
          state.failed_steps[step_id] = StepFailure(
            error_type=payload["error"]["error_type"],
            message=payload["error"]["message"],
            traceback=payload["error"].get("traceback"),
          )
          # Ensure a failed step doesn't also carry a stale success record.
          state.completed_steps.pop(step_id, None)

      return state

    return await asyncio.to_thread(_load)

  async def save_step_status(
    self, session_id: str, step_id: str, status: StepStatus
  ) -> None:
    """Persist a step status update without rewriting the entire session."""

    def _save() -> None:
      conn = self._ensure_connection()
      payload = json.dumps({"status": status.value})
      conn.execute(
        """
        INSERT INTO step_updates (session_id, step_id, update_type, payload_json)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(session_id, step_id, update_type) DO UPDATE SET
          payload_json = excluded.payload_json,
          updated_at = CURRENT_TIMESTAMP
        """,
        (session_id, step_id, "status", payload),
      )
      conn.commit()

    async with self._lock:
      await asyncio.to_thread(_save)

  async def save_step_result(
    self, session_id: str, step_id: str, result: Any
  ) -> None:
    """Persist a step result and mark it completed."""

    def _save() -> None:
      conn = self._ensure_connection()
      payload = json.dumps(
        {"status": StepStatus.COMPLETED.value, "result": result},
        default=str,
      )
      conn.execute(
        """
        INSERT INTO step_updates (session_id, step_id, update_type, payload_json)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(session_id, step_id, update_type) DO UPDATE SET
          payload_json = excluded.payload_json,
          updated_at = CURRENT_TIMESTAMP
        """,
        (session_id, step_id, "result", payload),
      )
      conn.commit()

    async with self._lock:
      await asyncio.to_thread(_save)

  async def save_step_error(
    self, session_id: str, step_id: str, error: Exception
  ) -> None:
    """Persist a step failure."""

    def _save() -> None:
      conn = self._ensure_connection()
      payload = json.dumps({
        "status": StepStatus.FAILED.value,
        "error": {
          "error_type": type(error).__name__,
          "message": str(error),
          "traceback": _traceback.format_exc() if _traceback else None,
        },
      })
      conn.execute(
        """
        INSERT INTO step_updates (session_id, step_id, update_type, payload_json)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(session_id, step_id, update_type) DO UPDATE SET
          payload_json = excluded.payload_json,
          updated_at = CURRENT_TIMESTAMP
        """,
        (session_id, step_id, "error", payload),
      )
      conn.commit()

    async with self._lock:
      await asyncio.to_thread(_save)
