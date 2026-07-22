from __future__ import annotations

import shutil
from pathlib import Path

from nonoka.ext.eval.models import EvalSample
from nonoka.ext.eval.sandbox import PythonSandbox
from nonoka.ext.coding.diagnostics import (
  VerificationReport,
  VerifierDiagnostic,
  VerifierDiagnosticCode,
)


class CodeChecker:
  def check(self, sample: EvalSample, root: Path) -> tuple[bool, str]:
    report = self.check_detailed(sample, root)
    return report.passed, report.message

  def check_detailed(self, sample: EvalSample, root: Path) -> VerificationReport:
    solution = root / "solution.py"
    if not solution.exists():
      return VerificationReport(
        False, "solution.py was not created",
        VerifierDiagnostic(VerifierDiagnosticCode.MISSING_SOLUTION, "solution.py was not created", "solution.py"),
      )
    tests = sample.metadata.get("tests")
    if tests is None:
      harness = str(sample.metadata.get("test", ""))
      entry_point = sample.metadata.get("entry_point")
      if entry_point:
        harness += f"\ncheck({entry_point})\n"
    else:
      harness = "\n".join(str(test) for test in tests) + "\n"
    harness_path = root / "_nonoka_eval_test.py"
    # A repair can overwrite solution.py several times within one filesystem
    # timestamp tick.  Remove bytecode produced by an earlier verifier so an
    # import never executes a stale, same-size candidate implementation.
    shutil.rmtree(root / "__pycache__", ignore_errors=True)
    harness_path.write_text("from solution import *\n" + harness, encoding="utf-8")
    result = PythonSandbox(root).run_file(harness_path)
    if result.timed_out:
      return VerificationReport(
        False, "verifier timed out",
        VerifierDiagnostic(VerifierDiagnosticCode.TIMEOUT, "verifier timed out", "solution.py"),
      )
    if result.returncode:
      detail = (result.stderr or result.stdout).strip()
      message = detail[-1000:] or f"verifier exited {result.returncode}"
      return VerificationReport(False, message, _diagnostic_for_failure(message))
    return VerificationReport(True, "functional tests passed")


def _diagnostic_for_failure(message: str) -> VerifierDiagnostic:
  lowered = message.lower()
  if "nameerror" in lowered:
    return VerifierDiagnostic(VerifierDiagnosticCode.MISSING_FUNCTION, message, "solution.py")
  if "typeerror" in lowered and ("argument" in lowered or "positional" in lowered):
    return VerifierDiagnostic(VerifierDiagnosticCode.SIGNATURE_MISMATCH, message, "solution.py")
  if "assertionerror" in lowered:
    return VerifierDiagnostic(VerifierDiagnosticCode.ASSERTION_FAILURE, message, "solution.py")
  if "filenotfounderror" in lowered or "permissionerror" in lowered:
    return VerifierDiagnostic(VerifierDiagnosticCode.WORKSPACE_FAILURE, message)
  return VerifierDiagnostic(VerifierDiagnosticCode.PROCESS_ERROR, message)
