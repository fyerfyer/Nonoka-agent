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
from nonoka.core.types import Capability


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
  external: bool = field(default=True, init=False)
  metadata: dict[str, Any] = field(default_factory=dict)

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
