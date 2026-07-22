"""Reproducible benchmark-matrix planning and execution.

The matrix records the exact model policy, task selection and harness for a
release candidate.  Framework-owned code tasks execute in-process; complex
benchmarks keep their official runners rather than being reimplemented here.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nonoka.ext.eval.datasets import get_registry
from nonoka.ext.eval.external import (
  export_evalplus_tasks,
  run_evalplus,
  run_tau2_bench,
  run_terminal_bench,
)
from nonoka.ext.eval.models import EvalRun, EvalSample
from nonoka.ext.eval.runners.headless import HeadlessEvalRunner

SCHEMA_VERSION = 1


def build_manifest(model: str, temperature: float, max_turns: int, timeout: float) -> dict[str, Any]:
  """Build the full release matrix without starting any model requests."""
  return {
    "schema_version": SCHEMA_VERSION,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "framework_revision": _git_revision(),
    "runtime": {"python": sys.version.split()[0], "platform": platform.platform()},
    "policy": {
      "model": model,
      "temperature": temperature,
      "max_turns": max_turns,
      "timeout_seconds": timeout,
      "trials": 1,
    },
    "gates": {
      "deterministic": (
        "Run core and eval adapter regression tests before spending model or Docker resources."
      ),
      "external": (
        "Run `nonoka eval doctor`; only then launch isolated official benchmark harnesses."
      ),
    },
    "jobs": [
      {
        "id": "humaneval", "runner": "framework", "dataset": "humaneval",
        "limit": None, "offset": 0, "baseline": True,
      },
      {
        "id": "mbpp-sanitized", "runner": "framework", "dataset": "mbpp",
        "limit": None, "offset": 0, "baseline": True,
      },
      {
        "id": "evalplus-humaneval", "runner": "official-external", "benchmark": "evalplus",
        "dataset": "humaneval", "limit": None,
        "note": "Generate all candidates, then run the official HumanEval+ verifier without base-test fallback.",
      },
      {
        "id": "evalplus-mbpp", "runner": "official-external", "benchmark": "evalplus",
        "dataset": "mbpp", "limit": None,
        "note": "Generate all candidates, then run the official MBPP+ verifier without base-test fallback.",
      },
      {
        "id": "tau3-retail", "runner": "official-external", "benchmark": "tau2-bench",
        "domain": "retail", "limit": None,
      },
      {
        "id": "tau3-airline", "runner": "official-external", "benchmark": "tau2-bench",
        "domain": "airline", "limit": None,
      },
      {
        "id": "terminal-bench", "runner": "official-external", "benchmark": "terminal-bench",
        "limit": None,
        "note": "Terminal-Bench 2 through Harbor; do not compare with legacy terminal-bench-core 0.1.1 scores.",
      },
    ],
  }


def write_manifest(manifest: dict[str, Any], path: Path) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


async def run_manifest(
  manifest: dict[str, Any], output_dir: Path, selected_jobs: set[str] | None = None,
) -> dict[str, Any]:
  """Run selected jobs and write an immutable result index.

  EvalPlus receives a full candidate JSONL and owns the final code score.
  """
  output_dir.mkdir(parents=True, exist_ok=True)
  policy = manifest["policy"]
  records: list[dict[str, Any]] = []
  for job in manifest["jobs"]:
    job_id = str(job["id"])
    if selected_jobs and job_id not in selected_jobs:
      records.append({"id": job_id, "status": "skipped", "reason": "not selected"})
      continue
    try:
      if job["runner"] == "framework":
        samples = _select_framework_samples(get_registry().load(job["dataset"], None), job)
        runner = HeadlessEvalRunner(
          policy["model"], max_turns=policy["max_turns"],
          timeout_seconds=policy["timeout_seconds"], temperature=policy["temperature"],
          strategy=job.get("strategy", "auto"),
          max_verifier_iterations=int(job.get("max_verifier_iterations", 2)),
        )
        results = await runner.evaluate_many(samples)
        baseline = await runner.evaluate_many(samples, baseline=True) if job.get("baseline") else []
        run = EvalRun(
          dataset=job["dataset"], model=policy["model"], samples=results, baseline_samples=baseline,
          source="matrix", metadata={"matrix_job": job_id, "policy": policy, "sample_ids": [s.id for s in samples]},
        )
        artifact = output_dir / f"{job_id}.json"
        run.write(artifact)
        records.append({"id": job_id, "status": "completed", "artifact": str(artifact), "summary": run.summary()})
      elif job["benchmark"] == "evalplus":
        if job.get("task_ids") is not None or job.get("limit") is not None or int(job.get("offset", 0)):
          raise RuntimeError(
            "Official EvalPlus verification requires one completion for every benchmark task; "
            "use a framework job with task_ids for a bounded diagnostic slice."
          )
        task_file = output_dir / f"{job_id}-tasks.jsonl"
        export_evalplus_tasks(job["dataset"], task_file)
        task_rows = _read_jsonl(task_file)
        samples = [
          EvalSample(
            id=row["task_id"], dataset=job_id, prompt=row["prompt"], kind="code",
            metadata={"skip_local_verifier": True, "entry_point": row.get("entry_point"), "source": "EvalPlus"},
          )
          for row in task_rows
        ]
        runner = HeadlessEvalRunner(
          policy["model"], max_turns=policy["max_turns"],
          timeout_seconds=policy["timeout_seconds"], temperature=policy["temperature"],
        )
        generated = await runner.evaluate_many(samples)
        candidates = output_dir / f"{job_id}-candidates.jsonl"
        _write_evalplus_candidates(candidates, generated)
        official_result = run_evalplus(job["dataset"], candidates)
        records.append({
          "id": job_id, "status": "completed", "artifact": str(official_result),
          "candidates": str(candidates), "generated": len(generated),
        })
      elif job["benchmark"] == "tau2-bench":
        artifact_dir = output_dir / job_id
        code = run_tau2_bench(
          policy["model"], job.get("limit"), job["domain"], policy["max_turns"] * 3,
          int(policy["timeout_seconds"]), artifact_dir,
        )
        records.append({"id": job_id, "status": "completed" if code == 0 else "failed", "returncode": code, "artifact": str(artifact_dir)})
      elif job["benchmark"] == "terminal-bench":
        artifact_dir = output_dir / job_id
        code = run_terminal_bench(
          policy["model"], job.get("limit"), output=artifact_dir,
          test_timeout_seconds=policy["timeout_seconds"] * 3,
        )
        records.append({"id": job_id, "status": "completed" if code == 0 else "failed", "returncode": code, "artifact": str(artifact_dir)})
    except Exception as exc:
      records.append({"id": job_id, "status": "blocked", "reason": f"{type(exc).__name__}: {exc}"})

  result = {
    "schema_version": SCHEMA_VERSION,
    "manifest": manifest,
    "started_at": datetime.now(timezone.utc).isoformat(),
    "records": records,
  }
  write_manifest(result, output_dir / "matrix-results.json")
  return result


def _git_revision() -> str:
  repo_root = Path(__file__).resolve().parents[3]
  try:
    return subprocess.run(
      ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, capture_output=True, check=True,
    ).stdout.strip()
  except (OSError, subprocess.SubprocessError):
    return "unknown"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
  return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _select_framework_samples(samples: list[EvalSample], job: dict[str, Any]) -> list[EvalSample]:
  """Apply reproducible bounded selection for framework-owned checkers."""
  task_ids = job.get("task_ids")
  if task_ids is not None:
    samples_by_id = {sample.id: sample for sample in samples}
    missing = [str(task_id) for task_id in task_ids if str(task_id) not in samples_by_id]
    if missing:
      raise RuntimeError(f"Unknown framework task IDs for {job['id']!r}: {', '.join(missing)}")
    return [samples_by_id[str(task_id)] for task_id in task_ids]
  offset = int(job.get("offset", 0))
  limit = job.get("limit")
  return samples[offset:] if limit is None else samples[offset:offset + int(limit)]


def _write_evalplus_candidates(path: Path, results: list[Any]) -> None:
  with path.open("w", encoding="utf-8") as handle:
    for result in results:
      handle.write(json.dumps({"task_id": result.sample_id, "solution": result.candidate_code or ""}) + "\n")
