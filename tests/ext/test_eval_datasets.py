from __future__ import annotations

from nonoka.ext.eval.datasets import get_registry
from nonoka.ext.eval.datasets.builtins import load_tool_use
from nonoka.ext.eval.models import EvalRun


def test_builtin_registry_lists_three_datasets():
  datasets = get_registry().list()
  assert [dataset.name for dataset in datasets] == ["humaneval", "mbpp", "tool_use"]


def test_tool_use_suite_is_versioned_and_limitable():
  all_samples = load_tool_use()
  limited = load_tool_use(3)
  assert len(all_samples) == 12
  assert len(limited) == 3
  assert len({sample.id for sample in all_samples}) == len(all_samples)
  assert all(sample.metadata["expected"] for sample in all_samples)
  assert all(sample.metadata["required_tools"] for sample in all_samples)


def test_registry_can_select_a_non_overlapping_window():
  samples = get_registry().load("tool_use", limit=3, offset=3)
  assert [sample.id for sample in samples] == [
    "tool_use/nested-config",
    "tool_use/grep-and-index",
    "tool_use/python-transform",
  ]


def test_run_summary_preserves_unknown_cost_as_null():
  run = EvalRun(dataset="tool_use", model="fake")
  assert run.summary()["total_estimated_cost_usd"] == 0.0
