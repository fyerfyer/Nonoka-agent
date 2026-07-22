"""Structured, deterministic verifier failures for coding workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class VerifierDiagnosticCode(str, Enum):
  MISSING_SOLUTION = "missing_solution"
  MISSING_FUNCTION = "missing_function"
  SIGNATURE_MISMATCH = "signature_mismatch"
  ASSERTION_FAILURE = "assertion_failure"
  TEST_FAILURE = "test_failure"
  WORKSPACE_FAILURE = "workspace_failure"
  TIMEOUT = "timeout"
  PROCESS_ERROR = "process_error"


@dataclass(frozen=True)
class VerifierDiagnostic:
  code: VerifierDiagnosticCode
  message: str
  path: str | None = None
  expected: str | None = None
  actual: str | None = None

  def to_dict(self) -> dict[str, Any]:
    return asdict(self) | {"code": self.code.value}


@dataclass(frozen=True)
class VerificationReport:
  passed: bool
  message: str
  diagnostic: VerifierDiagnostic | None = None

  @property
  def details(self) -> dict[str, Any]:
    return {"diagnostic": self.diagnostic.to_dict()} if self.diagnostic else {}
