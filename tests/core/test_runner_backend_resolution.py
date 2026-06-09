"""Tests for Runner backend resolution and configuration API."""

import pytest

from nonoka.core.runner import Runner
from nonoka.backends.checkpoint.sqlite import SQLiteCheckpointStore
from nonoka.backends.checkpoint.memory import MemoryCheckpointStore
from nonoka.backends.checkpoint.noop import NoOpCheckpointStore
from nonoka.backends.memory.sqlite import SQLiteMemoryBackend
from nonoka.backends.memory.in_memory import InMemoryBackend


# --------------------------------------------------------------------------- #
# Checkpoint resolution
# --------------------------------------------------------------------------- #

def test_runner_default_checkpoint_is_sqlite():
  """Runner() with no arguments should default to SQLiteCheckpointStore."""
  runner = Runner()
  assert isinstance(runner.checkpoint_store, SQLiteCheckpointStore)


def test_runner_checkpoint_memory():
  """Runner(checkpoint='memory') should create MemoryCheckpointStore."""
  runner = Runner(checkpoint="memory")
  assert isinstance(runner.checkpoint_store, MemoryCheckpointStore)


def test_runner_checkpoint_disabled():
  """Runner(checkpoint='disabled') should create NoOpCheckpointStore."""
  runner = Runner(checkpoint="disabled")
  assert isinstance(runner.checkpoint_store, NoOpCheckpointStore)


def test_runner_checkpoint_custom_object():
  """Runner(checkpoint=custom_obj) should accept a custom checkpoint store."""
  custom = SQLiteCheckpointStore(":memory:")
  runner = Runner(checkpoint=custom)
  assert runner.checkpoint_store is custom


def test_runner_checkpoint_invalid_string():
  """Runner(checkpoint='invalid') should raise ValueError."""
  with pytest.raises((TypeError, ValueError)):
    Runner(checkpoint="invalid")


# --------------------------------------------------------------------------- #
# Memory resolution
# --------------------------------------------------------------------------- #

def test_runner_default_memory_is_sqlite():
  """Runner() with no arguments should default to SQLiteMemoryBackend."""
  runner = Runner()
  assert isinstance(runner.memory_backend, SQLiteMemoryBackend)


def test_runner_memory_none():
  """Runner(memory=None) should not create any memory backend."""
  runner = Runner(memory=None)
  assert runner.memory_backend is None


def test_runner_memory_in_memory():
  """Runner(memory='in_memory') should create InMemoryBackend."""
  runner = Runner(memory="in_memory")
  assert isinstance(runner.memory_backend, InMemoryBackend)


def test_runner_memory_disabled():
  """Runner(memory='disabled') should not create any memory backend."""
  runner = Runner(memory="disabled")
  assert runner.memory_backend is None


def test_runner_memory_custom_object():
  """Runner(memory=custom_obj) should accept a custom memory backend."""
  custom = InMemoryBackend()
  runner = Runner(memory=custom)
  assert runner.memory_backend is custom


def test_runner_memory_invalid_string():
  """Runner(memory='invalid') should raise ValueError."""
  with pytest.raises((TypeError, ValueError)):
    Runner(memory="invalid")


# --------------------------------------------------------------------------- #
# Combined configurations
# --------------------------------------------------------------------------- #

def test_runner_all_defaults():
  """Default Runner should have SQLite for both checkpoint and memory."""
  runner = Runner()
  assert isinstance(runner.checkpoint_store, SQLiteCheckpointStore)
  assert isinstance(runner.memory_backend, SQLiteMemoryBackend)


def test_runner_memory_only():
  """Runner(checkpoint='memory', memory='in_memory') should use memory backends."""
  runner = Runner(checkpoint="memory", memory="in_memory")
  assert isinstance(runner.checkpoint_store, MemoryCheckpointStore)
  assert isinstance(runner.memory_backend, InMemoryBackend)


def test_runner_fully_disabled():
  """Runner(checkpoint='disabled', memory=None) should have no persistence."""
  runner = Runner(checkpoint="disabled", memory=None)
  assert isinstance(runner.checkpoint_store, NoOpCheckpointStore)
  assert runner.memory_backend is None


# --------------------------------------------------------------------------- #
# Duck typing validation
# --------------------------------------------------------------------------- #

def test_runner_checkpoint_duck_typing():
  """Runner should accept objects that match the CheckpointStore protocol."""
  class FakeStore:
    async def save_session(self, session_id, state): pass
    async def load_session(self, session_id): return None
    async def save_step_status(self, session_id, step_id, status): pass
    async def save_step_result(self, session_id, step_id, result): pass
    async def save_step_error(self, session_id, step_id, error): pass

  fake = FakeStore()
  runner = Runner(checkpoint=fake)
  assert runner.checkpoint_store is fake


def test_runner_checkpoint_duck_typing_missing_method():
  """Runner should reject objects missing CheckpointStore methods."""
  class BadStore:
    async def save_session(self, session_id, state): pass
    # Missing load_session and others

  with pytest.raises(TypeError):
    Runner(checkpoint=BadStore())


def test_runner_memory_duck_typing():
  """Runner should accept objects that match the MemoryBackend protocol."""
  class FakeBackend:
    async def add(self, content, session_id=None, user_id=None, metadata=None): pass
    async def search(self, query, session_id=None, user_id=None, limit=5): return []
    async def get_history(self, session_id, limit=None): return []
    async def get_user_memory(self, user_id, limit=10): return []

  fake = FakeBackend()
  runner = Runner(memory=fake)
  assert runner.memory_backend is fake


def test_runner_memory_duck_typing_missing_method():
  """Runner should reject objects missing MemoryBackend methods."""
  class BadBackend:
    async def add(self, content, session_id=None, user_id=None, metadata=None): pass
    # Missing search and others

  with pytest.raises(TypeError):
    Runner(memory=BadBackend())
