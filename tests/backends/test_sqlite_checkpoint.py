import asyncio
import os
import tempfile
from datetime import datetime

import pytest

from nonoka.backends.checkpoint.sqlite import SQLiteCheckpointStore
from nonoka.core.session import (
  SessionState,
  SessionStatus,
  StepStatus,
  StepResult,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture
async def memory_store():
  """Create a fresh in-memory store for each test."""
  store = SQLiteCheckpointStore(":memory:")
  yield store
  await store.close()


@pytest.fixture
async def file_store():
  """Create a fresh file-backed store for each test."""
  fd, path = tempfile.mkstemp(suffix=".db")
  os.close(fd)
  store = SQLiteCheckpointStore(path)
  yield store
  await store.close()
  os.unlink(path)


def _make_state(session_id: str = "sess-1", **kwargs) -> SessionState:
  """Helper to build a SessionState with sensible defaults."""
  return SessionState(
    session_id=session_id,
    status=SessionStatus.RUNNING,
    **kwargs,
  )


# --------------------------------------------------------------------------- #
# Core save / load
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_save_and_load_session(memory_store):
  """Saving a session and loading it back should return an equivalent state."""
  state = _make_state(
    session_id="sess-1",
    step_statuses={"step-1": StepStatus.COMPLETED},
    completed_steps={"step-1": StepResult(data={"key": "value"})},
    turn_count=5,
  )

  await memory_store.save_session("sess-1", state)
  loaded = await memory_store.load_session("sess-1")

  assert loaded is not None
  assert loaded.session_id == "sess-1"
  assert loaded.status == SessionStatus.RUNNING
  assert loaded.step_statuses == {"step-1": StepStatus.COMPLETED}
  assert loaded.completed_steps["step-1"].data == {"key": "value"}
  assert loaded.turn_count == 5


@pytest.mark.asyncio
async def test_load_nonexistent_session(memory_store):
  """Loading a session that was never saved should return None."""
  loaded = await memory_store.load_session("nonexistent")
  assert loaded is None


@pytest.mark.asyncio
async def test_overwrite_existing_session(memory_store):
  """Saving to the same session_id should overwrite the previous data."""
  state1 = _make_state(session_id="sess-1", turn_count=1)
  state2 = _make_state(session_id="sess-1", turn_count=99)

  await memory_store.save_session("sess-1", state1)
  await memory_store.save_session("sess-1", state2)

  loaded = await memory_store.load_session("sess-1")
  assert loaded is not None
  assert loaded.turn_count == 99


# --------------------------------------------------------------------------- #
# Step-level updates
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_save_step_status(memory_store):
  """save_step_status should be merged back into the loaded state."""
  state = _make_state(session_id="sess-1", step_statuses={})
  await memory_store.save_session("sess-1", state)

  await memory_store.save_step_status("sess-1", "step-a", StepStatus.RUNNING)
  loaded = await memory_store.load_session("sess-1")

  assert loaded is not None
  assert loaded.step_statuses["step-a"] == StepStatus.RUNNING


@pytest.mark.asyncio
async def test_save_step_result(memory_store):
  """save_step_result should mark the step as completed with data."""
  state = _make_state(session_id="sess-1")
  await memory_store.save_session("sess-1", state)

  await memory_store.save_step_result("sess-1", "step-a", result={"output": 42})
  loaded = await memory_store.load_session("sess-1")

  assert loaded is not None
  assert loaded.step_statuses["step-a"] == StepStatus.COMPLETED
  assert loaded.completed_steps["step-a"].data == {"output": 42}


@pytest.mark.asyncio
async def test_save_step_error(memory_store):
  """save_step_error should mark the step as failed with error details."""
  state = _make_state(session_id="sess-1")
  await memory_store.save_session("sess-1", state)

  try:
    raise ValueError("something went wrong")
  except ValueError as exc:
    await memory_store.save_step_error("sess-1", "step-a", error=exc)

  loaded = await memory_store.load_session("sess-1")

  assert loaded is not None
  assert loaded.step_statuses["step-a"] == StepStatus.FAILED
  assert loaded.failed_steps["step-a"].error_type == "ValueError"
  assert loaded.failed_steps["step-a"].message == "something went wrong"


@pytest.mark.asyncio
async def test_step_updates_persisted_independently(memory_store):
  """Step updates should be stored in a separate table and merged on load."""
  state = _make_state(session_id="sess-1", turn_count=0)
  await memory_store.save_session("sess-1", state)

  # Apply multiple step updates without re-saving the full session
  await memory_store.save_step_status("sess-1", "step-1", StepStatus.COMPLETED)
  await memory_store.save_step_result("sess-1", "step-1", result="done")
  await memory_store.save_step_status("sess-1", "step-2", StepStatus.FAILED)
  await memory_store.save_step_error("sess-1", "step-2", error=RuntimeError("oops"))

  loaded = await memory_store.load_session("sess-1")
  assert loaded is not None
  assert loaded.step_statuses["step-1"] == StepStatus.COMPLETED
  assert loaded.completed_steps["step-1"].data == "done"
  assert loaded.step_statuses["step-2"] == StepStatus.FAILED
  assert loaded.failed_steps["step-2"].error_type == "RuntimeError"


# --------------------------------------------------------------------------- #
# Complex data serialization
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_json_serialization_complex_data(memory_store):
  """SessionState with complex nested data should round-trip correctly."""
  state = _make_state(
    session_id="sess-complex",
    completed_steps={
      "step-1": StepResult(data={
        "nested": {"deep": [1, 2, 3]},
        "timestamp": datetime.now().isoformat(),
        "unicode": "你好世界 🌍",
      })
    },
    memory_entries=[
      {"role": "user", "content": "hello", "metadata": {"key": "val"}, "tokens": 10}
    ],
  )

  await memory_store.save_session("sess-complex", state)
  loaded = await memory_store.load_session("sess-complex")

  assert loaded is not None
  assert loaded.completed_steps["step-1"].data["nested"]["deep"] == [1, 2, 3]
  assert loaded.completed_steps["step-1"].data["unicode"] == "你好世界 🌍"
  assert len(loaded.memory_entries) == 1
  assert loaded.memory_entries[0]["content"] == "hello"


# --------------------------------------------------------------------------- #
# Isolation & concurrency
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_session_isolation(memory_store):
  """Sessions should be isolated from each other."""
  state_a = _make_state(session_id="sess-a", turn_count=1)
  state_b = _make_state(session_id="sess-b", turn_count=2)

  await memory_store.save_session("sess-a", state_a)
  await memory_store.save_session("sess-b", state_b)

  loaded_a = await memory_store.load_session("sess-a")
  loaded_b = await memory_store.load_session("sess-b")

  assert loaded_a.turn_count == 1
  assert loaded_b.turn_count == 2


@pytest.mark.asyncio
async def test_concurrent_access(memory_store):
  """Multiple concurrent save/load operations should not corrupt data."""
  state = _make_state(session_id="sess-concurrent")
  await memory_store.save_session("sess-concurrent", state)

  async def writer(step_id: str):
    await memory_store.save_step_result(
      "sess-concurrent", step_id, result=f"result-{step_id}"
    )

  # Launch 10 concurrent writes
  await asyncio.gather(*[writer(f"step-{i}") for i in range(10)])

  # Verify all writes are visible
  loaded = await memory_store.load_session("sess-concurrent")
  assert len(loaded.completed_steps) == 10
  for i in range(10):
    assert loaded.completed_steps[f"step-{i}"].data == f"result-step-{i}"


# --------------------------------------------------------------------------- #
# File persistence
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_persistence_across_instances(file_store):
  """Data written by one store instance should be readable by another."""
  state = _make_state(session_id="sess-persist", turn_count=42)
  await file_store.save_session("sess-persist", state)

  # Close and create a new instance pointing to the same file
  db_path = file_store._db_path
  await file_store.close()

  store2 = SQLiteCheckpointStore(db_path)
  loaded = await store2.load_session("sess-persist")

  assert loaded is not None
  assert loaded.turn_count == 42
  await store2.close()


# --------------------------------------------------------------------------- #
# Async context manager
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_async_context_manager():
  """Store should work as an async context manager."""
  async with SQLiteCheckpointStore(":memory:") as store:
    state = _make_state(session_id="sess-ctx")
    await store.save_session("sess-ctx", state)
    loaded = await store.load_session("sess-ctx")
    assert loaded is not None


# --------------------------------------------------------------------------- #
# Step update overwrite
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_step_update_overwrite(memory_store):
  """Saving a step update twice should keep the latest value."""
  state = _make_state(session_id="sess-1")
  await memory_store.save_session("sess-1", state)

  await memory_store.save_step_status("sess-1", "step-1", StepStatus.PENDING)
  await memory_store.save_step_status("sess-1", "step-1", StepStatus.COMPLETED)

  loaded = await memory_store.load_session("sess-1")
  assert loaded.step_statuses["step-1"] == StepStatus.COMPLETED


# --------------------------------------------------------------------------- #
# Load session without any step updates
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_load_session_without_step_updates(memory_store):
  """Loading a session that only has a full snapshot should work."""
  state = _make_state(
    session_id="sess-plain",
    step_statuses={"step-1": StepStatus.PENDING},
  )
  await memory_store.save_session("sess-plain", state)

  loaded = await memory_store.load_session("sess-plain")
  assert loaded is not None
  assert loaded.step_statuses["step-1"] == StepStatus.PENDING


# --------------------------------------------------------------------------- #
# Step status recovery ordering (Bug fix: P1.2)
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_load_session_step_status_not_overwritten_by_old_status(memory_store):
  """When a step completes, load_session should return COMPLETED even if
  an older 'status' row (e.g. RUNNING from step start) exists in step_updates.

  This verifies that step_updates are applied in chronological order so
  later updates (result / error) overwrite earlier ones (status)."""
  state = _make_state(
    session_id="sess-order",
    step_statuses={},
    completed_steps={},
  )
  await memory_store.save_session("sess-order", state)

  # Simulate step execution lifecycle:
  # 1. Step starts → save_step_status(RUNNING)
  await memory_store.save_step_status("sess-order", "step-1", StepStatus.RUNNING)
  # 2. Step completes → save_step_result(COMPLETED)
  await memory_store.save_step_result("sess-order", "step-1", result={"ok": True})

  loaded = await memory_store.load_session("sess-order")
  assert loaded is not None
  assert loaded.step_statuses["step-1"] == StepStatus.COMPLETED
  assert loaded.completed_steps["step-1"].data == {"ok": True}


@pytest.mark.asyncio
async def test_load_session_step_result_overrides_failed_status(memory_store):
  """A step that was previously marked FAILED should be updated to COMPLETED
  when a new result is saved (e.g. on retry-success)."""
  state = _make_state(
    session_id="sess-retry",
    step_statuses={},
    completed_steps={},
  )
  await memory_store.save_session("sess-retry", state)

  # 1. Step fails
  try:
    raise RuntimeError("boom")
  except RuntimeError as exc:
    await memory_store.save_step_error("sess-retry", "step-1", error=exc)
  # 2. Retry succeeds
  await memory_store.save_step_result("sess-retry", "step-1", result="fixed")

  loaded = await memory_store.load_session("sess-retry")
  assert loaded is not None
  assert loaded.step_statuses["step-1"] == StepStatus.COMPLETED
  assert loaded.completed_steps["step-1"].data == "fixed"
  assert "step-1" not in loaded.failed_steps
