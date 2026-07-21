from __future__ import annotations

import os
import gzip
import json

from nonoka.ext.eval.datasets import builtins
from nonoka.ext.eval.datasets.builtins import _compatible_proxy_environment


def test_huggingface_loader_temporarily_hides_only_socks_all_proxy(monkeypatch):
  monkeypatch.setenv("ALL_PROXY", "socks://127.0.0.1:7890/")
  monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7890/")
  with _compatible_proxy_environment():
    assert "ALL_PROXY" not in os.environ
    assert os.environ["HTTPS_PROXY"] == "http://127.0.0.1:7890/"
  assert os.environ["ALL_PROXY"] == "socks://127.0.0.1:7890/"


def test_humaneval_uses_official_jsonl_fallback_when_hub_loader_fails(monkeypatch):
  row = {"task_id": "HumanEval/0", "prompt": "def f():\n", "test": "", "entry_point": "f"}
  payload = gzip.compress((json.dumps(row) + "\n").encode())

  class Response:
    def read(self):
      return payload
    def __enter__(self):
      return self
    def __exit__(self, *_args):
      return None

  monkeypatch.setattr(builtins, "_load_hf", lambda *_args, **_kwargs: (_ for _ in ()).throw(builtins.DatasetLoaderError("offline")))
  monkeypatch.setattr(builtins.request.OpenerDirector, "open", lambda *_args, **_kwargs: Response())
  samples = builtins.load_humaneval()
  assert samples[0].id == "HumanEval/0"
  assert samples[0].metadata["entry_point"] == "f"


def test_mbpp_loader_passes_sanitized_configuration(monkeypatch):
  captured = {}

  def fake_loader(dataset_id, **kwargs):
    captured.update(dataset_id=dataset_id, **kwargs)
    return [{"task_id": 1, "text": "add", "test_list": ["assert add(1, 2) == 3"]}]

  monkeypatch.setattr(builtins, "_load_hf", fake_loader)
  samples = builtins.load_mbpp(1)
  assert captured == {"dataset_id": "google-research-datasets/mbpp", "name": "sanitized", "split": "test"}
  assert samples[0].metadata["entry_point"] == "add"
  assert "`add`" in samples[0].prompt


def test_mbpp_uses_official_json_fallback_when_hub_loader_fails(monkeypatch):
  payload = json.dumps([{"task_id": 2, "prompt": "add", "test_list": ["assert add(1, 2) == 3"]}]).encode()

  class Response:
    def read(self):
      return payload
    def __enter__(self):
      return self
    def __exit__(self, *_args):
      return None

  monkeypatch.setattr(builtins, "_load_hf", lambda *_args, **_kwargs: (_ for _ in ()).throw(builtins.DatasetLoaderError("offline")))
  monkeypatch.setattr(builtins.request.OpenerDirector, "open", lambda *_args, **_kwargs: Response())
  samples = builtins.load_mbpp()
  assert samples[0].id == "mbpp/2"
