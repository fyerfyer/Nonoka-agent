"""Authenticated FastAPI service backed by the public Runner API."""

from __future__ import annotations

import asyncio
import hmac
import json
import os
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from nonoka.core.agent import Agent
from nonoka.core.runner import Runner
from nonoka.observability import (
  ObservabilityPipeline,
  PostgresEventStore,
  SQLiteEventStore,
)


class RunRequest(BaseModel):
  prompt: str = Field(min_length=1)
  model: str | None = None
  session_id: str | None = None


class TaskInfo(BaseModel):
  task_id: str
  session_id: str
  status: str
  result: Any = None
  error: str | None = None


@dataclass
class _Task:
  info: TaskInfo
  future: asyncio.Task[Any]


def create_app(*, api_token: str | None = None, event_db: str | Path | None = None) -> FastAPI:
  """Create the service. Token is mandatory outside explicit test injection."""
  token = api_token if api_token is not None else os.getenv("NONOKA_API_TOKEN")
  if not token:
    raise RuntimeError("NONOKA_API_TOKEN is required to start the Nonoka service")
  database_url = os.getenv("NONOKA_EVENT_DATABASE_URL")
  store = (
    PostgresEventStore(database_url)
    if database_url else SQLiteEventStore(event_db or os.getenv("NONOKA_EVENT_DB", ".nonoka/nonoka-events.db"))
  )
  observability = ObservabilityPipeline(store)
  tasks: dict[str, _Task] = {}

  @asynccontextmanager
  async def lifespan(_: FastAPI):
    yield
    for task in list(tasks.values()):
      if not task.future.done():
        task.future.cancel()
    await asyncio.gather(*(task.future for task in tasks.values()), return_exceptions=True)
    await observability.close()

  app = FastAPI(title="nonoka-agent", version="1.3.8", lifespan=lifespan)

  async def authorize(authorization: str | None = Header(default=None)) -> None:
    if authorization is None or not hmac.compare_digest(authorization, f"Bearer {token}"):
      raise HTTPException(status_code=401, detail="invalid bearer token")

  def build_runner() -> Runner:
    return Runner(observability=observability)

  def build_agent(model: str | None) -> Agent:
    return Agent(model=model or os.getenv("NONOKA_DEFAULT_MODEL", "deepseek/deepseek-v4-pro"))

  async def execute(request: RunRequest) -> dict[str, Any]:
    agent = build_agent(request.model)
    result = await build_runner().run_react(
      agent, request.prompt, deps=None, session_id=request.session_id,
    )
    session_id = result.session.session_id if result.session else request.session_id or "unknown"
    summary = asdict(await store.summary(session_id))
    return {"session_id": session_id, "success": result.success, "data": result.data, "error": result.error, "error_type": result.error_type, "usage": summary, "trace": result.trace}

  @app.get("/health")
  async def health() -> dict[str, str]:
    return {"status": "ok"}

  @app.get("/metrics", response_class=PlainTextResponse)
  async def metrics() -> str:
    events = await store.list(limit=1000)
    return "\n".join([
      "# HELP nonoka_events_total Persisted execution events visible in the retention window.",
      "# TYPE nonoka_events_total gauge", f"nonoka_events_total {len(events)}",
      "# HELP nonoka_active_tasks Currently running asynchronous tasks.",
      "# TYPE nonoka_active_tasks gauge",
      f"nonoka_active_tasks {sum(not item.future.done() for item in tasks.values())}", "",
    ])

  @app.post("/run", dependencies=[Depends(authorize)])
  async def run(request: RunRequest) -> dict[str, Any]:
    return await execute(request)

  @app.post("/chat", dependencies=[Depends(authorize)])
  async def chat(request: RunRequest) -> StreamingResponse:
    async def events():
      try:
        async for event in build_runner().run_react_stream(
          build_agent(request.model), request.prompt, deps=None, session_id=request.session_id,
        ):
          yield f"event: {event.type}\ndata: {json.dumps(event.data, default=str)}\n\n"
      except Exception as exc:
        yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"
    return StreamingResponse(events(), media_type="text/event-stream")

  @app.post("/tasks", dependencies=[Depends(authorize)], response_model=TaskInfo)
  async def create_task(request: RunRequest) -> TaskInfo:
    task_id, session_id = str(uuid.uuid4()), request.session_id or str(uuid.uuid4())
    info = TaskInfo(task_id=task_id, session_id=session_id, status="running")
    async def worker() -> None:
      try:
        response = await execute(request.model_copy(update={"session_id": session_id}))
        info.status, info.result = ("completed" if response["success"] else "failed"), response
      except asyncio.CancelledError:
        info.status = "cancelled"
        raise
      except Exception as exc:
        info.status, info.error = "failed", str(exc)
    tasks[task_id] = _Task(info, asyncio.create_task(worker()))
    return info

  @app.get("/tasks", dependencies=[Depends(authorize)], response_model=list[TaskInfo])
  async def list_tasks() -> list[TaskInfo]:
    return [item.info for item in tasks.values()]

  @app.get("/tasks/{task_id}", dependencies=[Depends(authorize)], response_model=TaskInfo)
  async def get_task(task_id: str) -> TaskInfo:
    task = tasks.get(task_id)
    if task is None:
      raise HTTPException(status_code=404, detail="task not found")
    return task.info

  @app.delete("/tasks/{task_id}", dependencies=[Depends(authorize)], response_model=TaskInfo)
  async def cancel_task(task_id: str) -> TaskInfo:
    task = tasks.get(task_id)
    if task is None:
      raise HTTPException(status_code=404, detail="task not found")
    task.future.cancel()
    task.info.status = "cancelled"
    return task.info

  return app
