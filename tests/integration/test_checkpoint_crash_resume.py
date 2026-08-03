"""End-to-end crash-recovery tests against a file-backed SQLite checkpoint.

A Runner persists state through ``SQLiteCheckpointStore`` on disk, the
in-memory objects are discarded, and a brand-new Runner resumes from the
same database file.  No real LLM calls — providers are mocked.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from nonoka.backends.checkpoint.sqlite import SQLiteCheckpointStore
from nonoka.core.agent import Agent
from nonoka.core.agent_tool import AgentTool
from nonoka.core.context import RunContext
from nonoka.core.memory import MemoryRole
from nonoka.core.runner import Runner
from nonoka.core.session import SESSION_STATE_SCHEMA_VERSION, SessionStatus
from nonoka.core.tool import tool


def _make_runner(store: SQLiteCheckpointStore, provider: MagicMock) -> Runner:
  """Create a Runner on the given store with a mocked LLM provider."""
  runner = Runner(checkpoint=store, memory="in_memory")
  provider.chat_stream = AsyncMock(return_value=iter([]))
  runner._create_llm = lambda agent: provider  # type: ignore[method-assign]
  runner.llm = provider
  return runner


def _final_answer_provider(content: str = "final answer") -> MagicMock:
  provider = MagicMock()
  provider.chat = AsyncMock(
    return_value=MagicMock(content=content, tool_calls=None, usage={})
  )
  return provider


def _counting_tool() -> tuple[Any, dict[str, int]]:
  """A tool that records how many times it was invoked."""
  calls = {"count": 0}

  @tool
  async def probe(ctx: RunContext) -> str:
    calls["count"] += 1
    return "probe result"

  return probe, calls


def _read_state_json(db_path: Any, session_id: str) -> dict[str, Any]:
  """Read the raw checkpoint payload straight from the SQLite file."""
  conn = sqlite3.connect(str(db_path))
  try:
    row = conn.execute(
      "SELECT state_json FROM checkpoints WHERE session_id = ?",
      (session_id,),
    ).fetchone()
  finally:
    conn.close()
  assert row is not None
  return json.loads(row[0])


# --------------------------------------------------------------------------- #
# Scenario A — mid-turn crash replays dangling tool calls after a file roundtrip
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_midturn_crash_resume_through_sqlite_file(tmp_path):
  probe, calls = _counting_tool()
  agent = Agent(model="test", tools=[probe])
  db_path = tmp_path / "checkpoints.db"

  # Simulate a crash: the assistant tool_calls message was persisted through
  # the SQLite file, the tool result never was, and the process is gone.
  store1 = SQLiteCheckpointStore(str(db_path))
  runner1 = _make_runner(store1, _final_answer_provider())
  session = await runner1._create_session(agent, deps=None)
  await session.memory.add("run the probe", MemoryRole.USER)
  await session.memory.add(
    "",
    MemoryRole.ASSISTANT,
    tool_calls=[{
      "id": "call_1",
      "type": "function",
      "function": {"name": "probe", "arguments": "{}"},
    }],
  )
  session.status = SessionStatus.PAUSED
  await store1.save_session(session.session_id, session.to_state())
  session_id = session.session_id
  await store1.close()
  del session, runner1, store1

  # A fresh process resumes from the same database file.
  store2 = SQLiteCheckpointStore(str(db_path))
  runner2 = _make_runner(store2, _final_answer_provider())
  result = await runner2.resume(agent, session_id=session_id, deps=None)
  await store2.close()

  assert calls["count"] == 1
  assert result.success is True
  assert result.session.status == SessionStatus.COMPLETED
  tool_entries = [
    e for e in result.session.memory.entries if e.role == MemoryRole.TOOL
  ]
  assert len(tool_entries) == 1
  assert tool_entries[0].metadata["tool_call_id"] == "call_1"

  # The persisted JSON carries the schema version.
  payload = _read_state_json(db_path, session_id)
  assert payload["schema_version"] == SESSION_STATE_SCHEMA_VERSION


# --------------------------------------------------------------------------- #
# Scenario B — schema version enforcement on the file backend
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_newer_schema_version_rejected_and_legacy_payload_loads(tmp_path):
  db_path = tmp_path / "checkpoints.db"
  store = SQLiteCheckpointStore(str(db_path))
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
  legacy = {
    "session_id": "legacy",
    "status": SessionStatus.RUNNING.value,
    "turn_count": 2,
  }
  conn.execute(
    "INSERT INTO checkpoints (session_id, state_json) VALUES (?, ?)",
    ("legacy", json.dumps(legacy)),
  )
  conn.commit()
  await store.close()

  # Re-open through a fresh store so both payloads come from disk.
  store2 = SQLiteCheckpointStore(str(db_path))
  with pytest.raises(ValueError, match="newer version of nonoka"):
    await store2.load_session("future")

  loaded = await store2.load_session("legacy")
  await store2.close()

  assert loaded is not None
  assert loaded.schema_version == SESSION_STATE_SCHEMA_VERSION
  assert loaded.turn_count == 2


# --------------------------------------------------------------------------- #
# Scenario C — child-agent lineage survives a file roundtrip
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_agent_tool_lineage_resume_through_sqlite_file(tmp_path):
  child_agent = Agent(model="child", tools=[])
  parent_agent = Agent(model="parent", tools=[])
  at = AgentTool(agent=child_agent)
  db_path = tmp_path / "checkpoints.db"
  prompt = "do something"

  # Parent crash: a running lineage record and its orphaned child session
  # are both persisted through the SQLite file.
  store1 = SQLiteCheckpointStore(str(db_path))
  runner1 = _make_runner(store1, _final_answer_provider("child answer"))
  parent = await runner1._create_session(parent_agent, deps=None)
  orphan = await runner1._create_session(child_agent, deps=None)
  await orphan.memory.add("previous turn marker", MemoryRole.USER)
  orphan.status = SessionStatus.RUNNING
  await store1.save_session(orphan.session_id, orphan.to_state())
  parent.extension_state["agent_tool_lineage"] = {
    at._lineage_key(prompt): {
      "child_session_id": orphan.session_id,
      "status": "running",
      "result_text": None,
    }
  }
  await store1.save_session(parent.session_id, parent.to_state())
  parent_id = parent.session_id
  orphan_id = orphan.session_id
  await store1.close()
  del parent, orphan, runner1, store1

  # A fresh process restores the parent (lineage included) from disk and
  # re-invokes the sub-agent with the same prompt.
  store2 = SQLiteCheckpointStore(str(db_path))
  provider = _final_answer_provider("child answer")
  runner2 = _make_runner(store2, provider)
  parent2 = await runner2._create_session(parent_agent, deps=None, session_id=parent_id)

  result = await at.invoke(RunContext(parent2), {"task": prompt})
  await store2.close()

  assert result == "child answer"
  # The orphaned child session was continued, not recreated: its pre-existing
  # memory entry survived the roundtrip and reached the LLM.
  calls = provider.chat.call_args_list
  messages = calls[0].kwargs.get("messages") or calls[0][1].get("messages")
  all_content = " ".join(str(m.content) for m in messages)
  assert "previous turn marker" in all_content
  record = parent2.extension_state["agent_tool_lineage"][at._lineage_key(prompt)]
  assert record["child_session_id"] == orphan_id
  assert record["status"] == "completed"
