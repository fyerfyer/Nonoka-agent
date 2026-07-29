"""External tool capability.

An external capability is a tool whose actual execution is delegated to a host
or frontend (e.g. OpenCode). nonoka only registers the tool schema and emits the
tool call; the host executes it and returns the result via the resume path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from nonoka.core.context import RunContext
from nonoka.core.errors import ExternalToolExecutionRequiredError
from nonoka.core.execution import ToolExecution, UNKNOWN_EXECUTION
from nonoka.core.types import Capability


class ObservationCompleteness(str, Enum):
  """Host attestation describing whether an observation is exhaustive."""

  COMPLETE = "complete"
  PARTIAL = "partial"
  UNKNOWN = "unknown"


class VerificationStatus(str, Enum):
  """Outcome of a host-attested verification command."""

  PASSED = "passed"
  FAILED = "failed"
  UNAVAILABLE = "unavailable"
  NOT_RUN = "not_run"


class VerificationLevel(str, Enum):
  """Scope claimed by a verification command."""

  FOCUSED = "focused"
  FULL = "full"


class VerificationKind(str, Enum):
  """Broad, host-classified verification command family."""

  TEST = "test"
  BUILD = "build"
  LINT = "lint"
  TYPECHECK = "typecheck"
  CUSTOM = "custom"


@dataclass(frozen=True)
class VerificationReceipt:
  """Structured evidence produced by the host for a verification command."""

  status: VerificationStatus
  level: VerificationLevel = VerificationLevel.FOCUSED
  kind: VerificationKind = VerificationKind.CUSTOM
  command: str = ""
  cwd: str = ""
  exit_code: int | None = None
  timed_out: bool = False
  timeout_seconds: float | None = None
  truncated: bool = False
  completeness: ObservationCompleteness = ObservationCompleteness.UNKNOWN
  collected_tests: int | None = None
  summary: str | None = None
  failure_summary: str | None = None
  artifact_ref: str | None = None

  @classmethod
  def from_value(cls, value: "VerificationReceipt | dict[str, Any]") -> "VerificationReceipt":
    if isinstance(value, cls):
      return value
    if not isinstance(value, dict):
      raise TypeError("verification receipt must be a mapping")
    return cls(
      status=VerificationStatus(value.get("status", VerificationStatus.NOT_RUN)),
      level=VerificationLevel(value.get("level", VerificationLevel.FOCUSED)),
      kind=VerificationKind(value.get("kind", VerificationKind.CUSTOM)),
      command=str(value.get("command", "")),
      cwd=str(value.get("cwd", "")),
      exit_code=int(value["exit_code"]) if value.get("exit_code") is not None else None,
      timed_out=bool(value.get("timed_out", False)),
      timeout_seconds=(
        float(value["timeout_seconds"])
        if value.get("timeout_seconds") is not None else None
      ),
      truncated=bool(value.get("truncated", False)),
      completeness=ObservationCompleteness(
        value.get("completeness", ObservationCompleteness.UNKNOWN)
      ),
      collected_tests=(
        int(value["collected_tests"])
        if value.get("collected_tests") is not None else None
      ),
      summary=str(value["summary"]) if value.get("summary") is not None else None,
      failure_summary=(
        str(value["failure_summary"])
        if value.get("failure_summary") is not None else None
      ),
      artifact_ref=(
        str(value["artifact_ref"])
        if value.get("artifact_ref") is not None else None
      ),
    )


@dataclass(frozen=True)
class EffectAttestation:
  """Host declaration that an external action changed task-relevant state.

  ``scope`` is descriptive (for example ``workspace`` or ``system``), not a
  policy switch. ``changed`` is the only completion-relevant field. Keeping
  this typed prevents core from guessing state changes from tool names or
  shell commands while still supporting terminal tasks whose effects live
  outside the current working directory.
  """

  changed: bool
  scope: str = "external"
  collector: str = "host"
  summary: str | None = None

  @classmethod
  def from_value(cls, value: "EffectAttestation | dict[str, Any]") -> "EffectAttestation":
    if isinstance(value, cls):
      return value
    if not isinstance(value, dict):
      raise TypeError("effect attestation must be a mapping")
    return cls(
      changed=bool(value.get("changed", False)),
      scope=str(value.get("scope", "external")),
      collector=str(value.get("collector", "host")),
      summary=str(value["summary"]) if value.get("summary") is not None else None,
    )


@dataclass(frozen=True)
class WorkspaceAttestation:
  """A host-produced, content-addressed summary of an external workspace.

  This is an *attestation*, not a sandbox or a claim that an untrusted host
  cannot lie.  Its purpose is to make the trust boundary explicit and to give
  callers a uniform, auditable representation of effects that happened
  outside Nonoka's process.
  """

  root: str
  before_digest: str
  after_digest: str
  created: tuple[str, ...] = ()
  modified: tuple[str, ...] = ()
  deleted: tuple[str, ...] = ()
  policy_violations: tuple[str, ...] = ()
  restored_paths: tuple[str, ...] = ()
  collector: str = "host"

  @classmethod
  def from_value(cls, value: "WorkspaceAttestation | dict[str, Any]") -> "WorkspaceAttestation":
    if isinstance(value, cls):
      return value
    if not isinstance(value, dict):
      raise TypeError("workspace attestation must be a mapping")
    required = ("root", "before_digest", "after_digest")
    missing = [key for key in required if not isinstance(value.get(key), str) or not value[key]]
    if missing:
      raise ValueError(f"workspace attestation missing required fields: {', '.join(missing)}")
    return cls(
      root=value["root"], before_digest=value["before_digest"], after_digest=value["after_digest"],
      created=tuple(str(item) for item in value.get("created", ())),
      modified=tuple(str(item) for item in value.get("modified", ())),
      deleted=tuple(str(item) for item in value.get("deleted", ())),
      policy_violations=tuple(str(item) for item in value.get("policy_violations", ())),
      restored_paths=tuple(str(item) for item in value.get("restored_paths", ())),
      collector=str(value.get("collector", "host")),
    )


@dataclass(frozen=True)
class ExternalToolReceipt:
  """Result returned by a host after executing an :class:`ExternalCapability`.

  Legacy hosts may still return a raw result for capabilities without declared
  workspace mutation.  A capability marked ``mutates_workspace=True`` must
  return this receipt with a :class:`WorkspaceAttestation` before the session
  can be resumed.
  """

  result: Any = None
  exit_code: int | None = None
  elapsed_seconds: float | None = None
  workspace: WorkspaceAttestation | None = None
  effect: EffectAttestation | None = None
  host: str | None = None
  artifact_ref: str | None = None
  output_kind: str | None = None
  original_bytes: int | None = None
  truncated: bool = False
  completeness: ObservationCompleteness = ObservationCompleteness.UNKNOWN
  verification: VerificationReceipt | None = None

  def __post_init__(self) -> None:
    # Preserve explicitly supplied completeness, while mapping receipts from
    # older hosts that only exposed ``truncated`` onto the typed contract.
    if self.completeness == ObservationCompleteness.UNKNOWN and self.truncated:
      object.__setattr__(self, "completeness", ObservationCompleteness.PARTIAL)

  @classmethod
  def from_value(cls, value: "ExternalToolReceipt | dict[str, Any] | Any") -> "ExternalToolReceipt":
    if isinstance(value, cls):
      return value
    # ``{"result": ...}`` is a common ordinary tool payload.  Treat a
    # mapping as a receipt only when it carries host-execution metadata.
    receipt_keys = {
      "workspace", "effect", "exit_code", "elapsed_seconds", "host", "artifact_ref",
      "output_kind", "original_bytes", "truncated", "completeness",
      "verification",
    }
    if not isinstance(value, dict) or not receipt_keys & set(value):
      return cls(result=value)
    workspace = value.get("workspace")
    effect = value.get("effect")
    verification = value.get("verification")
    return cls(
      result=value.get("result"),
      exit_code=value.get("exit_code"),
      elapsed_seconds=value.get("elapsed_seconds"),
      workspace=WorkspaceAttestation.from_value(workspace) if workspace is not None else None,
      effect=EffectAttestation.from_value(effect) if effect is not None else None,
      host=str(value["host"]) if value.get("host") is not None else None,
      artifact_ref=str(value["artifact_ref"]) if value.get("artifact_ref") is not None else None,
      output_kind=str(value["output_kind"]) if value.get("output_kind") is not None else None,
      original_bytes=int(value["original_bytes"]) if value.get("original_bytes") is not None else None,
      truncated=bool(value.get("truncated", False)),
      completeness=ObservationCompleteness(value.get("completeness", ObservationCompleteness.UNKNOWN)),
      verification=(
        VerificationReceipt.from_value(verification)
        if verification is not None else None
      ),
    )


@dataclass
class ExternalCapability:
  """A capability whose execution is delegated to an external host.

  Implements the :class:`nonoka.core.types.Capability` protocol enough for the
  nonoka Agent to register the tool schema and emit a tool call. The
  ``external=True`` marker tells :class:`ReActAgent` to pause and let the host
  execute the tool instead of calling :meth:`invoke`.

  Args:
    name: Tool name exposed to the model.
    description: Tool description.
    parameters: JSON Schema for the tool's input parameters.
    metadata: Optional routing metadata for the host. Not sent to the LLM.
  """

  name: str
  description: str
  parameters: dict[str, Any]
  execution: ToolExecution = field(default_factory=lambda: UNKNOWN_EXECUTION)
  audit_required: bool | None = None
  external: bool = field(default=True, init=False)
  metadata: dict[str, Any] = field(default_factory=dict)

  @property
  def requires_workspace_attestation(self) -> bool:
    """Whether a host receipt must prove the declared workspace mutation."""
    return self.execution.mutates_workspace if self.audit_required is None else self.audit_required

  async def invoke(self, ctx: RunContext, arguments: dict[str, Any]) -> Any:
    """Must never be called; execution is delegated to the host."""
    raise ExternalToolExecutionRequiredError(
      tool_call_id="unknown",
      tool_name=self.name,
      arguments=arguments,
      message=(
        f"External tool '{self.name}' must be executed by the host, "
        "not by nonoka."
      ),
    )

  def to_json_schema(self) -> dict[str, Any]:
    """Return the OpenAI-compatible function schema."""
    return {
      "type": "function",
      "function": {
        "name": self.name,
        "description": self.description,
        "parameters": self.parameters,
      },
    }
