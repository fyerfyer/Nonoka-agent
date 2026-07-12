"""External MCP registry.

Host-managed MCP servers: the host owns server lifecycle and tool execution;
nonoka only registers the tool schemas and emits tool calls. Results are
returned via ``resume_external_tools()``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from nonoka.core.external_tool import ExternalCapability
from nonoka.core.types import Capability


def _sanitize_tool_name(name: str) -> str:
  """Return a provider-safe tool name.

  OpenAI function names must match ``^[a-zA-Z0-9_-]+$``. We replace namespace
  separators (``:``) with ``__`` and replace any remaining invalid characters
  with underscores. Double underscores are preserved as the namespace marker.
  """
  sanitized = name.replace(":", "__")
  sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", sanitized)
  return sanitized.strip("_")


@dataclass
class ExternalMCPToolDefinition:
  """A tool provided by an external MCP server."""

  name: str
  description: str
  parameters: dict[str, Any]


@dataclass
class ExternalMCPServer:
  """An external MCP server whose lifecycle is managed by the host."""

  name: str
  tools: list[ExternalMCPToolDefinition]
  description: str = ""


class ExternalMCPRegistry:
  """Manage tool schemas from one or more host-managed MCP servers.

  Tools are exposed to the model with a ``mcp__<server>__<tool>`` prefix so
  they do not collide with host native tools or internal capabilities.
  """

  def __init__(self, servers: list[ExternalMCPServer] | None = None):
    self._servers: dict[str, ExternalMCPServer] = {}
    self._capabilities: dict[str, Capability] = {}
    for server in servers or []:
      self.add_server(server)

  def add_server(self, server: ExternalMCPServer) -> None:
    """Register an external MCP server and its tools."""
    self._servers[server.name] = server
    for tool in server.tools:
      prefixed = _sanitize_tool_name(f"mcp__{server.name}__{tool.name}")
      self._capabilities[prefixed] = ExternalCapability(
        name=prefixed,
        description=tool.description,
        parameters=tool.parameters,
        metadata={
          "kind": "mcp_tool",
          "server": server.name,
          "original_name": tool.name,
        },
      )

  @property
  def servers(self) -> dict[str, ExternalMCPServer]:
    """Return registered servers keyed by name."""
    return dict(self._servers)

  def get_tools(self) -> list[Capability]:
    """Return all capabilities registered by this registry."""
    return list(self._capabilities.values())

  def build_registry_block(self) -> str:
    """Build a system-prompt block listing available external MCP servers."""
    if not self._servers:
      return ""
    lines = ["## Available MCP Servers (external)"]
    for server in self._servers.values():
      lines.append(f"- `{server.name}`: {len(server.tools)} tool(s)")
      for tool in server.tools:
        prefixed = _sanitize_tool_name(f"mcp__{server.name}__{tool.name}")
        lines.append(f"  - `{prefixed}`")
    return "\n".join(lines)
