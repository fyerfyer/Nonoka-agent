from __future__ import annotations

from pathlib import Path

from nonoka.ext.eval.models import EvalSample
from nonoka.ext.eval.sandbox import PythonSandbox


class CodeChecker:
  def check(self, sample: EvalSample, root: Path) -> tuple[bool, str]:
    solution = root / "solution.py"
    if not solution.exists():
      return False, "solution.py was not created"
    tests = sample.metadata.get("tests")
    if tests is None:
      harness = str(sample.metadata.get("test", ""))
      entry_point = sample.metadata.get("entry_point")
      if entry_point:
        harness += f"\ncheck({entry_point})\n"
    else:
      harness = "\n".join(str(test) for test in tests) + "\n"
    harness_path = root / "_nonoka_eval_test.py"
    harness_path.write_text("from solution import *\n" + harness, encoding="utf-8")
    result = PythonSandbox(root).run_file(harness_path)
    if result.timed_out:
      return False, "verifier timed out"
    if result.returncode:
      detail = (result.stderr or result.stdout).strip()
      return False, detail[-1000:] or f"verifier exited {result.returncode}"
    return True, "functional tests passed"
