import json

import pytest

from nonoka.backends.checkpoint.sqlite import SQLiteCheckpointStore
from nonoka.core.session import (
  SESSION_STATE_SCHEMA_VERSION,
  SessionState,
  SessionStatus,
  migrate_session_state_dict,
)


def test_migrate_passthrough_for_current_version():
  """Payloads at the current schema version are returned unchanged."""
  data = {"schema_version": SESSION_STATE_SCHEMA_VERSION, "session_id": "s1"}
  assert migrate_session_state_dict(data) is data


def test_migrate_treats_missing_version_as_v1():
  """Checkpoints written before schema versioning load as version 1."""
  data = {"session_id": "s1", "status": SessionStatus.RUNNING.value}
  migrated = migrate_session_state_dict(data)
  state = SessionState.model_validate(migrated)
  assert state.schema_version == SESSION_STATE_SCHEMA_VERSION
  assert state.session_id == "s1"


def test_migrate_rejects_newer_version():
  """A checkpoint written by a newer framework raises a clear error."""
  data = {"schema_version": SESSION_STATE_SCHEMA_VERSION + 1, "session_id": "s1"}
  with pytest.raises(ValueError, match="newer version of nonoka"):
    migrate_session_state_dict(data)


@pytest.mark.asyncio
async def test_sqlite_loads_legacy_checkpoint_without_version():
  """A raw legacy payload (no schema_version) loads through the SQLite store."""
  store = SQLiteCheckpointStore(":memory:")
  conn = store._ensure_connection()
  legacy = {
    "session_id": "legacy",
    "status": SessionStatus.RUNNING.value,
    "turn_count": 3,
  }
  conn.execute(
    "INSERT INTO checkpoints (session_id, state_json) VALUES (?, ?)",
    ("legacy", json.dumps(legacy)),
  )
  conn.commit()

  loaded = await store.load_session("legacy")
  await store.close()

  assert loaded is not None
  assert loaded.schema_version == SESSION_STATE_SCHEMA_VERSION
  assert loaded.turn_count == 3


@pytest.mark.asyncio
async def test_sqlite_rejects_checkpoint_from_newer_version():
  """Loading a checkpoint with a higher schema_version raises ValueError."""
  store = SQLiteCheckpointStore(":memory:")
  conn = store._ensure_connection()
  future = {
    "schema_version": SESSION_STATE_SCHEMA_VERSION + 1,
    "session_id": "future",
    "status": SessionStatus.RUNNING.value,
  }
  conn.execute(
    "INSERT INTO checkpoints (session_id, state_json) VALUES (?, ?)",
    ("future", json.dumps(future)),
  )
  conn.commit()

  with pytest.raises(ValueError, match="newer version of nonoka"):
    await store.load_session("future")
  await store.close()


@pytest.mark.asyncio
async def test_sqlite_roundtrip_preserves_schema_version():
  """Save/load keeps the current schema version on the state."""
  store = SQLiteCheckpointStore(":memory:")
  state = SessionState(session_id="rt", status=SessionStatus.RUNNING)

  await store.save_session("rt", state)
  loaded = await store.load_session("rt")
  await store.close()

  assert loaded is not None
  assert loaded.schema_version == SESSION_STATE_SCHEMA_VERSION
