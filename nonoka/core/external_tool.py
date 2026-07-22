"""External tool capability.

An external capability is a tool whose actual execution is delegated to a host
or frontend (e.g. OpenCode). nonoka only registers the tool schema and emits the
tool call; the host executes it and returns the result via the resume path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from nonoka.core.context import RunContext
from nonoka.core.errors import ExternalToolExecutionRequiredError
from nonoka.core.execution import ToolExecution, UNKNOWN_EXECUTION
from nonoka.core.types import Capability


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
  host: str | None = None

  @classmethod
  def from_value(cls, value: "ExternalToolReceipt | dict[str, Any] | Any") -> "ExternalToolReceipt":
    if isinstance(value, cls):
      return value
    # ``{"result": ...}`` is a common ordinary tool payload.  Treat a
    # mapping as a receipt only when it carries host-execution metadata.
    if not isinstance(value, dict) or not {"workspace", "exit_code", "elapsed_seconds", "host"} & set(value):
      return cls(result=value)
    workspace = value.get("workspace")
    return cls(
      result=value.get("result"),
      exit_code=value.get("exit_code"),
      elapsed_seconds=value.get("elapsed_seconds"),
      workspace=WorkspaceAttestation.from_value(workspace) if workspace is not None else None,
      host=str(value["host"]) if value.get("host") is not None else None,
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
