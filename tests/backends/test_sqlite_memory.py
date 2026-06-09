import os
import tempfile

import pytest

from nonoka.backends.memory.sqlite import SQLiteMemoryBackend


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture
async def memory_backend():
  """Create a fresh in-memory backend for each test."""
  backend = SQLiteMemoryBackend(":memory:")
  yield backend
  await backend.close()


@pytest.fixture
async def file_backend():
  """Create a fresh file-backed backend for each test."""
  fd, path = tempfile.mkstemp(suffix=".db")
  os.close(fd)
  backend = SQLiteMemoryBackend(path)
  yield backend
  await backend.close()
  os.unlink(path)


# --------------------------------------------------------------------------- #
# add / search
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_add_and_search(memory_backend):
  """Adding entries and searching should return matching results."""
  await memory_backend.add("User's favorite color is blue", session_id="sess-1")
  await memory_backend.add("User prefers warm weather", session_id="sess-1")
  await memory_backend.add("System configuration updated", session_id="sess-2")

  results = await memory_backend.search("color", session_id="sess-1")

  assert len(results) == 1
  assert results[0].content == "User's favorite color is blue"


@pytest.mark.asyncio
async def test_search_filters_by_session_id(memory_backend):
  """Search should only return entries matching the session_id filter."""
  await memory_backend.add("Session A message", session_id="sess-a")
  await memory_backend.add("Session B message", session_id="sess-b")
  await memory_backend.add("Another A message", session_id="sess-a")

  results_a = await memory_backend.search("message", session_id="sess-a")
  results_b = await memory_backend.search("message", session_id="sess-b")

  assert len(results_a) == 2
  assert len(results_b) == 1
  assert results_b[0].content == "Session B message"


@pytest.mark.asyncio
async def test_search_filters_by_user_id(memory_backend):
  """Search should only return entries matching the user_id filter."""
  await memory_backend.add("User 1 likes pizza", user_id="user-1")
  await memory_backend.add("User 2 likes sushi", user_id="user-2")
  await memory_backend.add("User 1 also likes pasta", user_id="user-1")

  results_1 = await memory_backend.search("likes", user_id="user-1")
  results_2 = await memory_backend.search("likes", user_id="user-2")

  assert len(results_1) == 2
  assert len(results_2) == 1
  assert results_2[0].content == "User 2 likes sushi"


@pytest.mark.asyncio
async def test_search_with_session_and_user_id(memory_backend):
  """Search should apply both session_id and user_id filters."""
  await memory_backend.add("Entry 1", session_id="sess-a", user_id="user-1")
  await memory_backend.add("Entry 2", session_id="sess-a", user_id="user-2")
  await memory_backend.add("Entry 3", session_id="sess-b", user_id="user-1")

  results = await memory_backend.search("Entry", session_id="sess-a", user_id="user-1")

  assert len(results) == 1
  assert results[0].content == "Entry 1"


@pytest.mark.asyncio
async def test_search_limit(memory_backend):
  """Search should respect the limit parameter."""
  for i in range(10):
    await memory_backend.add(f"Message {i}", session_id="sess-1")

  results = await memory_backend.search("Message", session_id="sess-1", limit=3)

  assert len(results) == 3


@pytest.mark.asyncio
async def test_search_returns_recent_first(memory_backend):
  """Search results should be ordered by most recent first."""
  await memory_backend.add("Older message", session_id="sess-1")
  await memory_backend.add("Newer message", session_id="sess-1")

  results = await memory_backend.search("message", session_id="sess-1")

  assert len(results) == 2
  assert results[0].content == "Newer message"
  assert results[1].content == "Older message"


@pytest.mark.asyncio
async def test_search_no_match(memory_backend):
  """Search with a query that doesn't match anything should return empty list."""
  await memory_backend.add("Hello world", session_id="sess-1")

  results = await memory_backend.search("nonexistent", session_id="sess-1")

  assert results == []


# --------------------------------------------------------------------------- #
# get_history
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_get_history(memory_backend):
  """get_history should return all entries for a session in chronological order."""
  await memory_backend.add("First", session_id="sess-1")
  await memory_backend.add("Second", session_id="sess-1")
  await memory_backend.add("Third", session_id="sess-1")
  await memory_backend.add("Other session", session_id="sess-2")

  history = await memory_backend.get_history("sess-1")

  assert len(history) == 3
  assert history[0].content == "First"
  assert history[1].content == "Second"
  assert history[2].content == "Third"


@pytest.mark.asyncio
async def test_get_history_with_limit(memory_backend):
  """get_history with limit should return the most recent N entries."""
  for i in range(5):
    await memory_backend.add(f"Message {i}", session_id="sess-1")

  history = await memory_backend.get_history("sess-1", limit=2)

  assert len(history) == 2
  assert history[0].content == "Message 3"
  assert history[1].content == "Message 4"


@pytest.mark.asyncio
async def test_get_history_no_entries(memory_backend):
  """get_history for a session with no entries should return empty list."""
  history = await memory_backend.get_history("nonexistent")

  assert history == []


# --------------------------------------------------------------------------- #
# get_user_memory
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_get_user_memory(memory_backend):
  """get_user_memory should return entries for a user, most recent first."""
  await memory_backend.add("Memory 1", user_id="user-1")
  await memory_backend.add("Memory 2", user_id="user-1")
  await memory_backend.add("Other user", user_id="user-2")

  memories = await memory_backend.get_user_memory("user-1")

  assert len(memories) == 2
  assert memories[0].content == "Memory 2"
  assert memories[1].content == "Memory 1"


@pytest.mark.asyncio
async def test_get_user_memory_with_limit(memory_backend):
  """get_user_memory should respect the limit parameter."""
  for i in range(5):
    await memory_backend.add(f"Memory {i}", user_id="user-1")

  memories = await memory_backend.get_user_memory("user-1", limit=2)

  assert len(memories) == 2
  assert memories[0].content == "Memory 4"
  assert memories[1].content == "Memory 3"


@pytest.mark.asyncio
async def test_get_user_memory_no_entries(memory_backend):
  """get_user_memory for a user with no entries should return empty list."""
  memories = await memory_backend.get_user_memory("nonexistent")

  assert memories == []


# --------------------------------------------------------------------------- #
# Metadata persistence
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_metadata_roundtrip(memory_backend):
  """Metadata dictionaries should survive the round-trip through SQLite."""
  await memory_backend.add(
    "Test content",
    session_id="sess-1",
    metadata={"key": "value", "number": 42, "nested": {"a": 1}},
  )

  results = await memory_backend.search("content", session_id="sess-1")

  assert len(results) == 1
  assert results[0].metadata["key"] == "value"
  assert results[0].metadata["number"] == 42
  assert results[0].metadata["nested"] == {"a": 1}


# --------------------------------------------------------------------------- #
# File persistence
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_persistence_across_instances(file_backend):
  """Data written by one backend instance should be readable by another."""
  await file_backend.add("Persistent memory", session_id="sess-1")

  # Close and create a new instance pointing to the same file
  db_path = file_backend._db_path
  await file_backend.close()

  backend2 = SQLiteMemoryBackend(db_path)
  results = await backend2.search("memory", session_id="sess-1")

  assert len(results) == 1
  assert results[0].content == "Persistent memory"
  await backend2.close()


# --------------------------------------------------------------------------- #
# Async context manager
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_async_context_manager():
  """Backend should work as an async context manager."""
  async with SQLiteMemoryBackend(":memory:") as backend:
    await backend.add("Context memory", session_id="sess-ctx")
    results = await backend.search("memory", session_id="sess-ctx")
    assert len(results) == 1


# --------------------------------------------------------------------------- #
# Session isolation
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_session_isolation(memory_backend):
  """Entries from different sessions should not leak into each other."""
  await memory_backend.add("Session A", session_id="sess-a")
  await memory_backend.add("Session B", session_id="sess-b")

  history_a = await memory_backend.get_history("sess-a")
  history_b = await memory_backend.get_history("sess-b")

  assert len(history_a) == 1
  assert len(history_b) == 1
  assert history_a[0].content == "Session A"
  assert history_b[0].content == "Session B"
