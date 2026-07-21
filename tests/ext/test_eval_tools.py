from __future__ import annotations

import asyncio

from nonoka.ext.eval.tools import EvalDeps, get_eval_tools


def test_execute_python_can_import_solution_from_isolated_workspace(tmp_path):
  (tmp_path / "solution.py").write_text("def answer():\n  return 42\n")
  execute_python = next(tool for tool in get_eval_tools() if tool.name == "execute_python")

  class Context:
    deps = EvalDeps(tmp_path)

  output = asyncio.run(execute_python.invoke(Context(), {"code": "from solution import answer; print(answer())"}))
  assert output["result"].strip() == "42"
