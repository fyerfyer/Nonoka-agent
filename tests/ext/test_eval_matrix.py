from __future__ import annotations

from nonoka.ext.eval.matrix import SCHEMA_VERSION, build_manifest


def test_default_matrix_is_full_and_pins_generation_policy():
  manifest = build_manifest("deepseek-chat", temperature=0.0, max_turns=8, timeout=90.0)

  assert manifest["schema_version"] == SCHEMA_VERSION
  assert manifest["policy"]["temperature"] == 0.0
  assert {job["id"] for job in manifest["jobs"]} == {
    "humaneval", "mbpp-sanitized", "evalplus-humaneval",
    "tau3-retail", "tau3-airline", "terminal-bench",
  }
  assert next(job for job in manifest["jobs"] if job["id"] == "humaneval")["limit"] is None
