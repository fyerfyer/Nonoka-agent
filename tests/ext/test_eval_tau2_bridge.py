from __future__ import annotations

import io
import json
import sys

from nonoka.ext.eval import tau2_adapter, tau2_bridge


def test_tau2_bridge_reuses_one_bridge_for_multiple_ndjson_requests(monkeypatch):
  instances = []

  class FakeBridge:
    def __init__(self):
      instances.append(self)

    async def respond(self, payload):
      return {"model": payload["model"], "content": "ok"}

  stdin = io.StringIO('{"model": "one", "messages": []}\n{"model": "two", "messages": []}\n')
  stdout = io.StringIO()
  monkeypatch.setattr(tau2_bridge, "_Bridge", FakeBridge)
  monkeypatch.setattr(tau2_bridge, "_load_environment", lambda: None)
  monkeypatch.setattr(sys, "stdin", stdin)
  monkeypatch.setattr(sys, "stdout", stdout)

  assert tau2_bridge._serve() == 0

  assert len(instances) == 1
  assert [json.loads(line)["model"] for line in stdout.getvalue().splitlines()] == ["one", "two"]


def test_persistent_tau2_client_sends_multiple_requests_to_one_process(monkeypatch):
  class FakeProcess:
    def __init__(self):
      self.stdin = io.StringIO()
      self.stdout = io.StringIO('{"content": "first"}\n{"content": "second"}\n')
      self.stderr = None

    def poll(self):
      return None

    def wait(self, timeout):
      return 0

  process = FakeProcess()
  calls = []
  monkeypatch.setattr(tau2_adapter.subprocess, "Popen", lambda *args, **kwargs: calls.append((args, kwargs)) or process)
  monkeypatch.setattr(tau2_adapter.select, "select", lambda readers, _write, _error, _timeout: (readers, [], []))

  client = tau2_adapter._NonokaBridgeClient("bridge-python")
  assert client.request({"model": "one"})["content"] == "first"
  assert client.request({"model": "two"})["content"] == "second"

  assert len(calls) == 1
  assert process.stdin.getvalue().splitlines() == ['{"model": "one"}', '{"model": "two"}']
