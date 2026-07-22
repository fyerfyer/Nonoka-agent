"""Versioned, deterministic MBPP slices used for strategy comparisons."""

from __future__ import annotations

import json
from pathlib import Path

from nonoka.ext.eval.datasets.base import DatasetLoaderError
from nonoka.ext.eval.datasets.builtins import load_mbpp
from nonoka.ext.eval.models import EvalSample

_FIXTURE_PATH = Path(__file__).with_name("fixtures") / "mbpp-complex-v1.json"


def _fixture() -> dict[str, object]:
  try:
    value = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover - packaging failure
    raise DatasetLoaderError("MBPP complex-v1 fixture is unavailable or invalid") from exc
  if not isinstance(value, dict) or value.get("version") != 1:
    raise DatasetLoaderError("MBPP complex-v1 fixture has an unsupported version")
  return value


def select_complex_mbpp_v1(samples: list[EvalSample]) -> list[EvalSample]:
  """Select the v1 candidates from an MBPP corpus using documented complexity signals.

  This helper is intentionally separate from the fixture-backed loader so an
  upstream corpus change is observable in a review instead of silently moving
  a benchmark target.
  """
  eligible = [sample for sample in samples if len(sample.metadata.get("tests", [])) >= 3]
  return sorted(
    eligible,
    key=lambda sample: (
      -len(sample.metadata.get("tests", [])),
      -sum(len(str(test)) for test in sample.metadata.get("tests", [])),
      -len(sample.prompt),
      sample.id,
    ),
  )[:20]


def load_complex_mbpp_v1(limit: int | None = None) -> list[EvalSample]:
  """Load the immutable MBPP complex-v1 task ordering."""
  fixture = _fixture()
  task_ids = fixture.get("task_ids")
  if not isinstance(task_ids, list) or not all(isinstance(task_id, str) for task_id in task_ids):
    raise DatasetLoaderError("MBPP complex-v1 fixture has invalid task IDs")
  samples_by_id = {sample.id: sample for sample in load_mbpp(None)}
  missing = [task_id for task_id in task_ids if task_id not in samples_by_id]
  if missing:
    raise DatasetLoaderError(
      "MBPP source does not contain every complex-v1 task: " + ", ".join(missing)
    )
  selected = [samples_by_id[task_id].model_copy(update={"dataset": "mbpp-complex-v1"}) for task_id in task_ids]
  return selected if limit is None else selected[:limit]
