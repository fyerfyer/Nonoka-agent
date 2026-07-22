from __future__ import annotations

from nonoka.ext.eval.datasets import get_registry
from nonoka.ext.eval.datasets.builtins import load_tool_use
from nonoka.ext.eval.datasets.complex_mbpp import load_complex_mbpp_v1, select_complex_mbpp_v1
from nonoka.ext.eval.models import EvalRun, EvalSample


def test_builtin_registry_lists_versioned_complex_mbpp_dataset():
  datasets = get_registry().list()
  assert [dataset.name for dataset in datasets] == ["humaneval", "mbpp", "mbpp-complex-v1", "tool_use"]


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


def test_complex_mbpp_selection_is_deterministic_and_fixture_backed(monkeypatch):
  samples = [
    EvalSample(
      id=f"mbpp/{number}", dataset="mbpp", kind="code", prompt="x" * prompt_size,
      metadata={"tests": ["a" * test_size] * test_count},
    )
    for number, test_count, test_size, prompt_size in [
      ("a", 3, 2, 5), ("b", 4, 1, 1), ("c", 3, 3, 1), ("d", 3, 3, 4),
    ]
  ]
  assert [sample.id for sample in select_complex_mbpp_v1(samples)] == ["mbpp/b", "mbpp/d", "mbpp/c", "mbpp/a"]

  fixture_samples = [
    EvalSample(id=f"mbpp/{task_id}", dataset="mbpp", kind="code", prompt="", metadata={"tests": []})
    for task_id in [799, 6, 756, 803, 172, 735, 802, 759, 758, 730, 725, 754, 773, 723, 766, 800, 721, 791, 726, 223]
  ]
  monkeypatch.setattr("nonoka.ext.eval.datasets.complex_mbpp.load_mbpp", lambda _limit: fixture_samples)

  selected = load_complex_mbpp_v1()
  assert [sample.id for sample in selected] == [sample.id for sample in fixture_samples]
  assert {sample.dataset for sample in selected} == {"mbpp-complex-v1"}
