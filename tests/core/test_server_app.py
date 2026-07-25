from __future__ import annotations

from fastapi.testclient import TestClient

from nonoka.core.runner import StreamEvent
from nonoka.core.session import Session
from nonoka.core.types import RunResult
from nonoka.server import app as server_app


class FakeRunner:
  def __init__(self, **kwargs):
    pass

  async def run_react(self, agent, prompt, deps, session_id=None):
    session = Session(session_id=session_id or "generated", agent=agent, deps=deps)
    return RunResult(success=True, data=f"echo:{prompt}", session=session)

  async def run_react_stream(self, agent, prompt, deps, session_id=None):
    yield StreamEvent(type="content_delta", data={"delta": "hello"})
    yield StreamEvent(type="final", data={"success": True, "data": "hello"})


def test_service_health_auth_run_metrics_and_chat(tmp_path, monkeypatch):
  monkeypatch.setattr(server_app, "Runner", FakeRunner)
  app = server_app.create_app(api_token="test-token", event_db=tmp_path / "events.db")

  with TestClient(app) as client:
    assert client.get("/health").json() == {"status": "ok"}
    assert client.post("/run", json={"prompt": "hello"}).status_code == 401

    headers = {"Authorization": "Bearer test-token"}
    response = client.post("/run", headers=headers, json={"prompt": "hello", "model": "test"})
    assert response.status_code == 200
    assert response.json()["data"] == "echo:hello"

    chat = client.post("/chat", headers=headers, json={"prompt": "hello", "model": "test"})
    assert chat.status_code == 200
    assert "event: content_delta" in chat.text
    assert "event: final" in chat.text

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert "# TYPE nonoka_events_total gauge" in metrics.text
    assert "# TYPE nonoka_active_tasks gauge" in metrics.text
