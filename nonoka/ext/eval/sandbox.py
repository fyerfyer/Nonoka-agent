"""Small, deterministic verifier subprocess wrapper.

This is filesystem isolation, not a general security sandbox.  Untrusted or
large workloads must use the Docker-backed external benchmarks instead.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SandboxResult:
  returncode: int
  stdout: str
  stderr: str
  timed_out: bool = False


class PythonSandbox:
  def __init__(self, root: Path, timeout_seconds: float = 10.0) -> None:
    self.root = root
    self.timeout_seconds = timeout_seconds

  def run_file(self, path: Path) -> SandboxResult:
    # ``-I`` deliberately removes the current directory from sys.path. Add
    # only this ephemeral workspace back so verifier code can import the
    # candidate ``solution.py`` without seeing the caller's environment.
    script = (
      "import runpy,sys; "
      f"sys.path.insert(0, {str(self.root)!r}); "
      f"runpy.run_path({str(path)!r}, run_name='__main__')"
    )
    try:
      result = subprocess.run(
        [sys.executable, "-I", "-c", script], cwd=self.root, text=True,
        capture_output=True, timeout=self.timeout_seconds, check=False,
      )
      return SandboxResult(result.returncode, result.stdout, result.stderr)
    except subprocess.TimeoutExpired as exc:
      return SandboxResult(124, exc.stdout or "", exc.stderr or "", timed_out=True)
