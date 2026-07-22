from __future__ import annotations

import asyncio

from nonoka.ext.eval import matrix
from nonoka.ext.eval.matrix import SCHEMA_VERSION, build_manifest
from nonoka.ext.eval.models import EvalSample


def test_default_matrix_is_full_and_pins_generation_policy():
  manifest = build_manifest("deepseek-chat", temperature=0.0, max_turns=8, timeout=90.0)

  assert manifest["schema_version"] == SCHEMA_VERSION
  assert manifest["policy"]["temperature"] == 0.0
  assert {job["id"] for job in manifest["jobs"]} == {
    "humaneval", "mbpp-sanitized", "evalplus-humaneval", "evalplus-mbpp",
    "tau3-retail", "tau3-airline", "terminal-bench",
  }
  assert next(job for job in manifest["jobs"] if job["id"] == "humaneval")["limit"] is None


def test_framework_matrix_honors_limit_and_offset():
  samples = [EvalSample(id=f"mbpp/{index}", dataset="mbpp", prompt="", kind="code") for index in range(5)]

  selected = matrix._select_framework_samples(samples, {"id": "slice", "offset": 2, "limit": 2})

  assert [sample.id for sample in selected] == ["mbpp/2", "mbpp/3"]


def test_framework_matrix_honors_explicit_task_ids():
  samples = [EvalSample(id=f"mbpp/{index}", dataset="mbpp", prompt="", kind="code") for index in range(3)]

  selected = matrix._select_framework_samples(
    samples, {"id": "selected", "offset": 1, "limit": 1, "task_ids": ["mbpp/2", "mbpp/0"]},
  )

  assert [sample.id for sample in selected] == ["mbpp/2", "mbpp/0"]


def test_evalplus_matrix_rejects_partial_task_selection(tmp_path):
  manifest = {
    "policy": {"model": "test", "max_turns": 1, "timeout_seconds": 1, "temperature": 0},
    "jobs": [{
      "id": "evalplus-slice", "runner": "official-external", "benchmark": "evalplus",
      "dataset": "mbpp", "limit": 2,
    }],
  }

  result = asyncio.run(matrix.run_manifest(manifest, tmp_path))

  assert result["records"][0]["status"] == "blocked"
  assert "requires one completion for every benchmark task" in result["records"][0]["reason"]
