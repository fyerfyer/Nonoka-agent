import pytest

from nonoka.core.agent import Agent
from nonoka.core.context import RunContext
from nonoka.core.session import Session
from nonoka.core.types import RunResult
from nonoka.observability import (
  ObservabilityHooks,
  ObservabilityPipeline,
  SQLiteEventStore,
)


@pytest.mark.asyncio
async def test_event_store_summarizes_usage_and_redacts_secrets(tmp_path):
  store = SQLiteEventStore(tmp_path / "events.db")
  await store.append("s1", "llm.response", {
    "api_key": "sk-secret-value-that-must-not-be-visible",
    "usage": {"prompt_tokens": 2, "completion_tokens": 3, "estimated_cost_usd": 0.1},
  })
  await store.append("s1", "tool.started", {"tool_name": "read"})
  events = await store.list("s1")
  usage = await store.summary("s1")
  assert events[0]["payload"]["api_key"] == "[REDACTED]"
  assert usage.total_tokens == 5
  assert usage.tool_calls == 1


@pytest.mark.asyncio
async def test_event_store_separates_cache_savings_from_actual_spend(tmp_path):
  store = SQLiteEventStore(tmp_path / "events.db")
  await store.append("s1", "llm.response", {
    "usage": {"prompt_tokens": 100, "completion_tokens": 20, "estimated_cost_usd": 0.5},
  })
  await store.append("s1", "llm.usage", {
    "usage": {"prompt_tokens": 100, "completion_tokens": 20, "estimated_cost_usd": 0.5, "cache_hit": True},
  })
  await store.append("s1", "llm.usage", {
    "usage": {"prompt_tokens": 3, "completion_tokens": 2, "estimated_cost_usd": 0.1, "cache_hit": False},
  })

  usage = await store.summary("s1")

  assert usage.llm_calls == 1
  assert usage.total_tokens == 5
  assert usage.estimated_cost_usd == 0.1
  assert usage.cache_hits == 1
  assert usage.cache_saved_tokens == 120
  assert usage.cache_saved_cost_usd == 0.5
  assert usage.estimated_cost_usd == pytest.approx(0.1)
  await store.close()


@pytest.mark.asyncio
async def test_pipeline_exports_redacted_event_without_failing_run(tmp_path):
  delivered = []

  class Exporter:
    async def emit(self, event):
      delivered.append(event)
    async def flush(self):
      pass
    async def close(self):
      pass

  pipeline = ObservabilityPipeline(SQLiteEventStore(tmp_path / "events.db"), [Exporter()])
  await pipeline.append("s1", "llm.request", {"authorization": "Bearer secret"})
  assert delivered[0].payload["authorization"] == "[REDACTED]"
  await pipeline.close()


@pytest.mark.asyncio
async def test_observability_hooks_write_error_event(tmp_path):
  store = SQLiteEventStore(tmp_path / "events.db")
  hooks = ObservabilityHooks(store)
  session = Session(session_id="s1", agent=Agent(model="test"), deps=None)
  ctx = RunContext(session)
  await hooks.on_session_start(ctx)
  await hooks.on_session_end(ctx, RunResult(success=False, session=session, error="failed", error_type="SafetyError"))
  assert [event["event_type"] for event in await store.list("s1")] == ["run.started", "run.finished", "error"]
  await store.close()


@pytest.mark.asyncio
async def test_observability_hooks_record_child_agent_identity(tmp_path):
  store = SQLiteEventStore(tmp_path / "events.db")
  hooks = ObservabilityHooks(store)
  agent = Agent(
    model="child-model",
    metadata={"project_agent_role": "reviewer"},
    tags=["subagent", "project-defined"],
  )
  session = Session(session_id="child", agent=agent, deps=None)
  object.__setattr__(session, "_parent_session_id", "parent")

  await hooks.on_session_start(RunContext(session))

  event = (await store.list("child"))[0]
  assert event["payload"] == {
    "model": "child-model",
    "metadata": {"project_agent_role": "reviewer"},
    "tags": ["subagent", "project-defined"],
    "parent_session_id": "parent",
  }
  await store.close()


@pytest.mark.asyncio
async def test_event_store_reads_existing_read_only_database(tmp_path):
  path = tmp_path / "events.db"
  writer = SQLiteEventStore(path)
  await writer.append("s1", "run.started", {"model": "test"})
  await writer.close()

  path.chmod(0o444)
  reader = SQLiteEventStore(path)
  assert (await reader.list("s1"))[0]["event_type"] == "run.started"
  await reader.close()
