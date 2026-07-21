from __future__ import annotations

from nonoka.ext.eval.checkers import CodeChecker, ToolUseChecker
from nonoka.ext.eval.datasets.builtins import load_tool_use
from nonoka.ext.eval.models import EvalSample
from nonoka.ext.eval.tools import EvalDeps


def test_code_checker_runs_functional_harness(tmp_path):
  sample = EvalSample(
    id="code/one", dataset="test", prompt="", kind="code",
    metadata={"tests": ["assert add(1, 2) == 3"]},
  )
  (tmp_path / "solution.py").write_text("def add(a, b):\n  return a + b\n")
  assert CodeChecker().check(sample, tmp_path) == (True, "functional tests passed")


def test_tool_use_checker_requires_trace_and_exact_workspace(tmp_path):
  sample = load_tool_use(1)[0]
  for name, content in sample.metadata["expected"].items():
    (tmp_path / name).write_text(content)
  deps = EvalDeps(
    tmp_path,
    tool_trace=["read_file:incoming.txt", "write_file:cleaned.txt", "execute_python"],
  )
  assert ToolUseChecker().check(sample, tmp_path, deps)[0] is True
  missing_read = EvalDeps(tmp_path, tool_trace=["write_file:cleaned.txt", "execute_python"])
  assert "missing required tool use" in ToolUseChecker().check(sample, tmp_path, missing_read)[1]
  assert ToolUseChecker().check(sample, tmp_path, EvalDeps(tmp_path))[0] is False
