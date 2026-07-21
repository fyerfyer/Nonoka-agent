"""Open dataset loaders plus a small, versioned local tool-use suite."""

from __future__ import annotations

import re
import os
import gzip
import io
import json
from urllib import request
from contextlib import contextmanager
from typing import Any

from nonoka.ext.eval.datasets.base import DatasetLoaderError
from nonoka.ext.eval.models import EvalSample


def _limit(samples: list[EvalSample], limit: int | None) -> list[EvalSample]:
  return samples if limit is None else samples[:limit]


@contextmanager
def _compatible_proxy_environment():
  """Prefer HTTP(S) proxy variables when a client lacks SOCKS support.

  Many developer environments set both an HTTP proxy and a SOCKS
  ``ALL_PROXY``. Hugging Face's transport can reject the latter without its
  optional SOCKS extra. Hide it only during this request and restore the user
  environment immediately afterwards.
  """
  saved = {key: os.environ.get(key) for key in ("ALL_PROXY", "all_proxy")}
  for key, value in saved.items():
    if value and value.lower().startswith("socks://"):
      os.environ.pop(key, None)
  try:
    yield
  finally:
    for key, value in saved.items():
      if value is not None:
        os.environ[key] = value


def _load_hf(dataset_id: str, **kwargs: Any) -> Any:
  try:
    from datasets import load_dataset
  except ImportError as exc:
    raise DatasetLoaderError("Install nonoka[eval] to load Hugging Face datasets.") from exc
  try:
    with _compatible_proxy_environment():
      return load_dataset(dataset_id, **kwargs)
  except Exception as exc:
    raise DatasetLoaderError(
      f"Could not load '{dataset_id}'. Check network/cache access or use tool_use: {exc}"
    ) from exc


def load_humaneval(limit: int | None = None) -> list[EvalSample]:
  try:
    dataset = _load_hf("openai/openai_humaneval", split="test")
    rows = list(dataset)
    source = "openai/openai_humaneval"
  except DatasetLoaderError:
    rows = _load_humaneval_github()
    source = _HUMANEVAL_GITHUB_URL
  samples = [
    EvalSample(
      id=str(row["task_id"]), dataset="humaneval", kind="code", prompt=str(row["prompt"]),
      metadata={"test": str(row["test"]), "entry_point": str(row["entry_point"]), "source": source},
    )
    for row in rows
  ]
  return _limit(samples, limit)


_HUMANEVAL_GITHUB_URL = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"


def _load_humaneval_github() -> list[dict[str, Any]]:
  """Load the official JSONL fallback without inheriting a SOCKS ALL_PROXY."""
  proxies = {
    scheme: value
    for scheme, value in {
      "http": os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy"),
      "https": os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy"),
    }.items()
    if value
  }
  opener = request.build_opener(request.ProxyHandler(proxies))
  try:
    with opener.open(_HUMANEVAL_GITHUB_URL, timeout=30) as response:
      payload = response.read()
  except Exception as exc:
    raise DatasetLoaderError(
      "Could not load HumanEval from Hugging Face or the official GitHub fallback. "
      f"Check proxy/network access: {exc}"
    ) from exc
  try:
    stream = io.TextIOWrapper(gzip.GzipFile(fileobj=io.BytesIO(payload)), encoding="utf-8")
    return [json.loads(line) for line in stream if line.strip()]
  except Exception as exc:
    raise DatasetLoaderError(f"HumanEval fallback returned invalid JSONL: {exc}") from exc


def load_mbpp(limit: int | None = None) -> list[EvalSample]:
  try:
    dataset = _load_hf("google-research-datasets/mbpp", name="sanitized", split="test")
    rows = list(dataset)
    source = "google-research-datasets/mbpp"
  except DatasetLoaderError:
    rows = _load_mbpp_github()
    source = _MBPP_GITHUB_URL
  samples: list[EvalSample] = []
  for row in rows:
    tests = [str(test) for test in row.get("test_list", [])]
    entry_point = _entry_point(tests)
    prompt = str(row.get("text") or row.get("prompt") or "")
    if entry_point:
      prompt += f"\n\nImplement the function named `{entry_point}` exactly as tested."
    samples.append(EvalSample(
      id=f"mbpp/{row['task_id']}", dataset="mbpp", kind="code",
      prompt=prompt,
      metadata={"tests": tests, "entry_point": entry_point, "source": source},
    ))
  return _limit(samples, limit)


_MBPP_GITHUB_URL = "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/sanitized-mbpp.json"


def _load_mbpp_github() -> list[dict[str, Any]]:
  proxies = {
    scheme: value
    for scheme, value in {
      "http": os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy"),
      "https": os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy"),
    }.items()
    if value
  }
  opener = request.build_opener(request.ProxyHandler(proxies))
  try:
    with opener.open(_MBPP_GITHUB_URL, timeout=30) as response:
      payload = response.read()
    rows = json.loads(payload.decode("utf-8"))
  except Exception as exc:
    raise DatasetLoaderError(
      "Could not load MBPP from Hugging Face or the official Google Research fallback. "
      f"Check proxy/network access: {exc}"
    ) from exc
  if not isinstance(rows, list):
    raise DatasetLoaderError("MBPP fallback returned a non-list JSON payload")
  return rows


def _entry_point(tests: list[str]) -> str | None:
  for test in tests:
    match = re.search(r"(?:assert\s+)?([A-Za-z_]\w*)\s*\(", test)
    if match:
      return match.group(1)
  return None


_TOOL_TASKS: tuple[dict[str, Any], ...] = (
  {"id": "rename-and-report", "prompt": "Read incoming.txt, replace every 'todo' with 'done', write cleaned.txt, then run a command proving cleaned.txt has exactly 3 lines.", "files": {"incoming.txt": "todo one\ntodo two\ntodo three\n"}, "expected": {"cleaned.txt": "done one\ndone two\ndone three\n"}},
  {"id": "json-summary", "prompt": "Read orders.json. Create summary.txt containing the total quantity and total revenue as `quantity=<n> revenue=<n>` and verify it with a command.", "files": {"orders.json": '[{"quantity":2,"price":3},{"quantity":4,"price":5}]'}, "expected": {"summary.txt": "quantity=6 revenue=26"}},
  {"id": "csv-filter", "prompt": "Read users.csv. Write active.txt with active user names sorted alphabetically, one per line, and check the output.", "files": {"users.csv": "name,active\nzoe,false\nanna,true\nmike,true\n"}, "expected": {"active.txt": "anna\nmike\n"}},
  {"id": "nested-config", "prompt": "Inspect config/settings.ini, change mode from development to production, save it, and use a command to show the resulting mode.", "files": {"config/settings.ini": "mode=development\nretries=2\n"}, "expected": {"config/settings.ini": "mode=production\nretries=2\n"}},
  {"id": "grep-and-index", "prompt": "Search notes.md for lines beginning with '- '. Create index.txt with their text without the dash, preserving order, then verify its line count.", "files": {"notes.md": "# Notes\n- alpha\nparagraph\n- beta\n- gamma\n"}, "expected": {"index.txt": "alpha\nbeta\ngamma\n"}},
  {"id": "python-transform", "prompt": "Read values.txt. Create squares.txt containing the square of each integer on its own line. Use a command to execute and verify the transformation.", "files": {"values.txt": "2\n3\n5\n"}, "expected": {"squares.txt": "4\n9\n25\n"}},
  {"id": "delete-temp", "prompt": "Remove obsolete.tmp, add archive.txt containing `archived`, and run a directory listing to verify both changes.", "files": {"obsolete.tmp": "discard\n"}, "expected": {"archive.txt": "archived\n"}, "absent": ["obsolete.tmp"], "required_tools": ["delete_file", "write_file", "list_dir"]},
  {"id": "patch-readme", "prompt": "Read README.md and CHANGELOG.md. Update README.md so its version matches the version in CHANGELOG.md, then verify the new line with grep.", "files": {"README.md": "Project\nVersion: 0.1.0\n", "CHANGELOG.md": "## 0.2.0\n- feature\n"}, "expected": {"README.md": "Project\nVersion: 0.2.0\n"}},
  {"id": "merge-fragments", "prompt": "Combine a.txt and b.txt into combined.txt, deduplicating lines while preserving first appearance, then count it with a command.", "files": {"a.txt": "red\nblue\n", "b.txt": "blue\ngreen\n"}, "expected": {"combined.txt": "red\nblue\ngreen\n"}},
  {"id": "find-source", "prompt": "Search src for the value `legacy`. Replace it with `current` only in Python files, leave text files unchanged, and verify with grep.", "files": {"src/app.py": "VALUE = 'legacy'\n", "src/readme.txt": "legacy\n"}, "expected": {"src/app.py": "VALUE = 'current'\n", "src/readme.txt": "legacy\n"}},
  {"id": "structured-log", "prompt": "Read events.log and create errors.txt containing only ERROR lines in their original order. Verify that no INFO line remains.", "files": {"events.log": "INFO boot\nERROR disk\nINFO retry\nERROR network\n"}, "expected": {"errors.txt": "ERROR disk\nERROR network\n"}},
  {"id": "inventory", "prompt": "Read inventory.json and write low-stock.txt with names whose stock is below 5, sorted alphabetically. Use a command to verify it.", "files": {"inventory.json": '[{"name":"wire","stock":3},{"name":"adapter","stock":9},{"name":"battery","stock":1}]'}, "expected": {"low-stock.txt": "battery\nwire\n"}},
)


def load_tool_use(limit: int | None = None) -> list[EvalSample]:
  samples = [
    EvalSample(
      id=f"tool_use/{task['id']}", dataset="tool_use", kind="tool_use", prompt=task["prompt"],
      metadata={
        **{k: v for k, v in task.items() if k not in {"id", "prompt"}},
        # A correct final file alone is insufficient: this benchmark checks
        # the multi-step interaction that the prompt asks the agent to perform.
        "required_tools": task.get("required_tools", ["read_file", "write_file", "execute_python"]),
      },
    )
    for task in _TOOL_TASKS
  ]
  return _limit(samples, limit)
