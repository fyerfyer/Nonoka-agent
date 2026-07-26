"""Structured, credential-redacted run events and usage aggregation."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from nonoka.core.hooks import Hooks
from nonoka.core.trace import redact


def _now() -> str:
  return datetime.now(timezone.utc).isoformat()


@dataclass
class UsageSummary:
  input_tokens: int = 0
  output_tokens: int = 0
  llm_calls: int = 0
  tool_calls: int = 0
  estimated_cost_usd: float | None = 0.0
  cache_hits: int = 0
  cache_saved_tokens: int = 0
  cache_saved_cost_usd: float | None = 0.0

  @property
  def total_tokens(self) -> int:
    return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class RunEvent:
  session_id: str
  event_type: str
  payload: dict[str, Any]
  occurred_at: str = ""


class TelemetryExporter(Protocol):
  """Optional third-party export boundary (Langfuse, Phoenix, custom OTLP)."""

  async def emit(self, event: RunEvent) -> None: ...
  async def flush(self) -> None: ...
  async def close(self) -> None: ...


class EventStore(Protocol):
  async def append(self, session_id: str, event_type: str, payload: dict[str, Any]) -> None: ...
  async def list(self, session_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]: ...
  async def summary(self, session_id: str) -> UsageSummary: ...
  async def close(self) -> None: ...


class ObservabilityPipeline:
  """Fan out redacted events while keeping external exporters best-effort."""

  def __init__(self, store: EventStore, exporters: list[TelemetryExporter] | None = None) -> None:
    self.store = store
    self.exporters = list(exporters or [])

  async def append(self, session_id: str, event_type: str, payload: dict[str, Any]) -> None:
    safe = redact(payload, max_chars=65536)
    await self.store.append(session_id, event_type, safe)
    event = RunEvent(session_id, event_type, safe, _now())
    for exporter in self.exporters:
      try:
        await exporter.emit(event)
      except Exception:
        # Optional telemetry must never fail an agent run.
        continue

  async def list(self, session_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    return await self.store.list(session_id, limit)

  async def summary(self, session_id: str) -> UsageSummary:
    return await self.store.summary(session_id)

  async def flush(self) -> None:
    for exporter in self.exporters:
      try:
        await exporter.flush()
      except Exception:
        continue

  async def close(self) -> None:
    await self.flush()
    for exporter in self.exporters:
      try:
        await exporter.close()
      except Exception:
        continue
    await self.store.close()


class SQLiteEventStore:
  """Small local event store shared by the CLI and server development mode."""

  def __init__(self, db_path: str | Path, retention_days: int = 30) -> None:
    self.db_path = str(db_path)
    self.retention_days = max(1, retention_days)
    self._conn: sqlite3.Connection | None = None
    self._read_only = False
    self._lock = asyncio.Lock()

  def _connection(self) -> sqlite3.Connection:
    if self._conn is None:
      path = Path(self.db_path).expanduser()
      path.parent.mkdir(parents=True, exist_ok=True)
      self._conn = sqlite3.connect(str(path), check_same_thread=False)
      self._conn.row_factory = sqlite3.Row
      try:
        self._conn.execute("""
          CREATE TABLE IF NOT EXISTS run_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            occurred_at TEXT NOT NULL,
            event_type TEXT NOT NULL,
            payload_json TEXT NOT NULL
          )
        """)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_run_events_session ON run_events(session_id, id)")
        self._conn.commit()
      except sqlite3.OperationalError as exc:
        if "readonly" not in str(exc).lower() or not path.exists():
          raise
        self._conn.close()
        # immutable avoids journal recovery writes when logs are inspected from
        # a read-only container or a different user namespace.
        self._conn = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._read_only = True
    return self._conn

  async def append(self, session_id: str, event_type: str, payload: dict[str, Any]) -> None:
    safe = redact(payload, max_chars=65536)
    def write() -> None:
      conn = self._connection()
      if self._read_only:
        raise OSError(f"Event store is read-only: {self.db_path}")
      conn.execute(
        "INSERT INTO run_events(session_id, occurred_at, event_type, payload_json) VALUES (?, ?, ?, ?)",
        (session_id, _now(), event_type, json.dumps(safe, ensure_ascii=False, default=str)),
      )
      conn.execute("DELETE FROM run_events WHERE occurred_at < datetime('now', ?)", (f"-{self.retention_days} days",))
      conn.commit()
    async with self._lock:
      write()

  async def list(self, session_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    limit = max(1, min(limit, 1000))
    def read() -> list[dict[str, Any]]:
      conn = self._connection()
      if session_id:
        rows = conn.execute("SELECT * FROM run_events WHERE session_id = ? ORDER BY id DESC LIMIT ?", (session_id, limit)).fetchall()
      else:
        rows = conn.execute("SELECT * FROM run_events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
      return [{"id": row["id"], "session_id": row["session_id"], "occurred_at": row["occurred_at"], "event_type": row["event_type"], "payload": json.loads(row["payload_json"])} for row in reversed(rows)]
    async with self._lock:
      return read()

  async def summary(self, session_id: str) -> UsageSummary:
    return _summary_from_events(await self.list(session_id, limit=1000))

  async def close(self) -> None:
    if self._conn is not None:
      self._conn.close()
      self._conn = None


class PostgresEventStore:
  """PostgreSQL event store selected by service deployments via a DSN."""

  def __init__(self, dsn: str, retention_days: int = 30) -> None:
    self.dsn = dsn
    self.retention_days = max(1, retention_days)
    self._conn: Any | None = None
    self._lock = asyncio.Lock()

  async def _connection(self) -> Any:
    if self._conn is None:
      import psycopg
      self._conn = await psycopg.AsyncConnection.connect(self.dsn)
      async with self._conn.cursor() as cursor:
        await cursor.execute("""
          CREATE TABLE IF NOT EXISTS nonoka_run_events (
            id BIGSERIAL PRIMARY KEY, session_id TEXT NOT NULL,
            occurred_at TIMESTAMPTZ NOT NULL, event_type TEXT NOT NULL,
            payload_json JSONB NOT NULL
          )
        """)
        await cursor.execute("CREATE INDEX IF NOT EXISTS idx_nonoka_events_session ON nonoka_run_events(session_id, id)")
      await self._conn.commit()
    return self._conn

  async def append(self, session_id: str, event_type: str, payload: dict[str, Any]) -> None:
    safe = redact(payload, max_chars=65536)
    async with self._lock:
      conn = await self._connection()
      async with conn.cursor() as cursor:
        await cursor.execute(
          "INSERT INTO nonoka_run_events(session_id, occurred_at, event_type, payload_json) VALUES (%s, NOW(), %s, %s::jsonb)",
          (session_id, event_type, json.dumps(safe, ensure_ascii=False, default=str)),
        )
        await cursor.execute("DELETE FROM nonoka_run_events WHERE occurred_at < NOW() - (%s || ' days')::interval", (str(self.retention_days),))
      await conn.commit()

  async def list(self, session_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    limit = max(1, min(limit, 1000))
    async with self._lock:
      conn = await self._connection()
      async with conn.cursor() as cursor:
        if session_id:
          await cursor.execute("SELECT id, session_id, occurred_at, event_type, payload_json FROM nonoka_run_events WHERE session_id=%s ORDER BY id DESC LIMIT %s", (session_id, limit))
        else:
          await cursor.execute("SELECT id, session_id, occurred_at, event_type, payload_json FROM nonoka_run_events ORDER BY id DESC LIMIT %s", (limit,))
        rows = await cursor.fetchall()
    return [{"id": row[0], "session_id": row[1], "occurred_at": row[2].isoformat(), "event_type": row[3], "payload": row[4]} for row in reversed(rows)]

  async def summary(self, session_id: str) -> UsageSummary:
    return _summary_from_events(await self.list(session_id, limit=1000))

  async def close(self) -> None:
    if self._conn is not None:
      await self._conn.close()
      self._conn = None


def _summary_from_events(events: list[dict[str, Any]]) -> UsageSummary:
  summary = UsageSummary()
  # ``llm.usage`` is emitted after cache normalization and is therefore the
  # authoritative cost ledger.  Keep response-event fallback for databases
  # written by older Nonoka versions.
  has_usage_events = any(event["event_type"] == "llm.usage" for event in events)
  for event in events:
    payload = event["payload"]
    if event["event_type"] == "llm.usage" or (not has_usage_events and event["event_type"] == "llm.response"):
      usage = payload.get("usage") if event["event_type"] == "llm.usage" else payload.get("usage")
      usage = usage or {}
      if usage.get("cache_hit"):
        summary.cache_hits += 1
        summary.cache_saved_tokens += int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        summary.cache_saved_tokens += int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        cost = usage.get("estimated_cost_usd")
        if cost is None:
          summary.cache_saved_cost_usd = None
        elif summary.cache_saved_cost_usd is not None:
          summary.cache_saved_cost_usd += float(cost)
        continue
      summary.llm_calls += 1
      summary.input_tokens += int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
      summary.output_tokens += int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
      cost = usage.get("estimated_cost_usd")
      if cost is None:
        summary.estimated_cost_usd = None
      elif summary.estimated_cost_usd is not None:
        summary.estimated_cost_usd += float(cost)
    elif event["event_type"] == "tool.started":
      summary.tool_calls += 1
  return summary


class ObservabilityHooks(Hooks):
  """Persist execution events using public Hooks without affecting execution."""

  def __init__(self, store: EventStore) -> None:
    super().__init__()
    self.store = store
    try:
      from opentelemetry import trace
      self._tracer = trace.get_tracer("nonoka")
    except Exception:
      self._tracer = None

  def _start_span(self, ctx: Any, name: str, attributes: dict[str, Any]) -> None:
    """Start a manual span so it covers the corresponding hook lifecycle."""
    extras = getattr(ctx, "extra", None)
    if self._tracer is None or extras is None:
      return
    span = self._tracer.start_span(name)
    for key, value in attributes.items():
      if value is not None:
        span.set_attribute(key, value)
    extras.setdefault("nonoka.observability.spans", {}).setdefault(name, []).append(span)

  def _end_span(self, ctx: Any, name: str, error: Exception | None = None) -> None:
    spans = getattr(ctx, "extra", {}).get("nonoka.observability.spans", {}).get(name, [])
    if not spans:
      return
    span = spans.pop()
    if error is not None:
      span.set_attribute("error.type", type(error).__name__)
      span.set_attribute("error.message", str(error))
    span.end()

  async def on_session_start(self, ctx) -> None:
    self._start_span(ctx, "nonoka.run", {
      "nonoka.session_id": ctx.session.session_id,
      "gen_ai.request.model": ctx.agent.model,
    })
    await self.store.append(ctx.session.session_id, "run.started", {"model": ctx.agent.model})

  async def on_session_end(self, ctx, result) -> None:
    await self.store.append(ctx.session.session_id, "run.finished", {
      "success": result.success, "error": result.error, "error_type": result.error_type,
      "usage": asdict(await self.store.summary(ctx.session.session_id)),
    })
    if result.error:
      await self.store.append(ctx.session.session_id, "error", {
        "message": result.error, "error_type": result.error_type,
      })
    self._end_span(ctx, "nonoka.run", RuntimeError(result.error) if result.error else None)

  async def on_llm_request(self, ctx, messages, tools) -> None:
    self._start_span(ctx, "nonoka.llm", {"gen_ai.request.model": ctx.agent.model})
    await self.store.append(ctx.session.session_id, "llm.request", {"messages": [m.model_dump(exclude_none=True) for m in messages], "tools": tools or []})

  async def on_llm_response(self, ctx, response) -> None:
    await self.store.append(ctx.session.session_id, "llm.response", {"content": response.content, "tool_calls": response.tool_calls, "usage": response.usage or {}})
    self._end_span(ctx, "nonoka.llm")

  async def on_llm_usage(self, ctx, usage) -> None:
    """Persist the post-cache accounting record without prompt content."""
    await self.store.append(ctx.session.session_id, "llm.usage", {"usage": usage})

  async def on_tool_start(self, ctx, tool_name, arguments) -> None:
    span_name = f"nonoka.tool.{tool_name}"
    self._start_span(ctx, span_name, {"gen_ai.tool.name": tool_name})
    await self.store.append(ctx.session.session_id, "tool.started", {"tool_name": tool_name, "arguments": arguments})

  async def on_tool_end(self, ctx, tool_name, arguments, result, error) -> None:
    await self.store.append(ctx.session.session_id, "tool.finished", {"tool_name": tool_name, "arguments": arguments, "result": result, "error": str(error) if error else None})
    self._end_span(ctx, f"nonoka.tool.{tool_name}", error)
