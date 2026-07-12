"""Tests for host-managed external MCP registry."""

from __future__ import annotations

import pytest

from nonoka import (
  AgentBuilder,
  ExternalMCPRegistry,
  ExternalMCPServer,
  ExternalMCPToolDefinition,
)
from nonoka.core.errors import ExternalToolExecutionRequiredError
from nonoka.core.external_mcp import _sanitize_tool_name


def test_sanitize_tool_name():
  assert _sanitize_tool_name("list_directory") == "list_directory"
  assert _sanitize_tool_name("mcp:filesystem:list_directory") == "mcp__filesystem__list_directory"
  # Hyphens are allowed in OpenAI function names; dots are not.
  assert _sanitize_tool_name("tool-with.dots") == "tool-with_dots"


def test_registry_add_server_creates_prefixed_capabilities():
  registry = ExternalMCPRegistry()
  registry.add_server(
    ExternalMCPServer(
      name="filesystem",
      description="Filesystem access",
      tools=[
        ExternalMCPToolDefinition(
          name="list_directory",
          description="List files in a directory.",
          parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
          },
        ),
      ],
    )
  )

  tools = registry.get_tools()
  assert len(tools) == 1
  cap = tools[0]
  assert cap.name == "mcp__filesystem__list_directory"
  assert cap.description == "List files in a directory."
  assert cap.external is True
  assert cap.metadata == {
    "kind": "mcp_tool",
    "server": "filesystem",
    "original_name": "list_directory",
  }

  schema = cap.to_json_schema()
  assert schema["function"]["name"] == "mcp__filesystem__list_directory"
  assert "metadata" not in schema["function"]


@pytest.mark.asyncio
async def test_external_mcp_capability_invoke_raises():
  registry = ExternalMCPRegistry([
    ExternalMCPServer(
      name="fs",
      tools=[
        ExternalMCPToolDefinition(
          name="read",
          description="Read a file.",
          parameters={"type": "object", "properties": {}},
        ),
      ],
    ),
  ])
  cap = registry.get_tools()[0]
  with pytest.raises(ExternalToolExecutionRequiredError):
    await cap.invoke(None, {"path": "/tmp"})


def test_registry_block_includes_servers_and_tools():
  registry = ExternalMCPRegistry([
    ExternalMCPServer(
      name="filesystem",
      tools=[
        ExternalMCPToolDefinition("list_directory", "List", {}),
        ExternalMCPToolDefinition("read_file", "Read", {}),
      ],
    ),
    ExternalMCPServer(
      name="fetch",
      tools=[ExternalMCPToolDefinition("fetch_url", "Fetch", {})],
    ),
  ])

  block = registry.build_registry_block()
  assert "## Available MCP Servers (external)" in block
  assert "`filesystem`: 2 tool(s)" in block
  assert "`mcp__filesystem__list_directory`" in block
  assert "`mcp__filesystem__read_file`" in block
  assert "`fetch`: 1 tool(s)" in block
  assert "`mcp__fetch__fetch_url`" in block


def test_empty_registry_returns_empty_tools_and_block():
  registry = ExternalMCPRegistry()
  assert registry.get_tools() == []
  assert registry.build_registry_block() == ""


def test_later_server_tool_overwrites_earlier():
  registry = ExternalMCPRegistry([
    ExternalMCPServer(
      name="a",
      tools=[ExternalMCPToolDefinition("tool", "from a", {})],
    ),
    ExternalMCPServer(
      name="b",
      tools=[ExternalMCPToolDefinition("tool", "from b", {})],
    ),
  ])
  # Same original tool name from different servers gets different prefixed names.
  names = {cap.name for cap in registry.get_tools()}
  assert names == {"mcp__a__tool", "mcp__b__tool"}


def test_agent_builder_external_mcp_registry():
  registry = ExternalMCPRegistry([
    ExternalMCPServer(
      name="fs",
      tools=[ExternalMCPToolDefinition("list", "List", {})],
    ),
  ])
  agent = (
    AgentBuilder()
    .model("gpt-4o")
    .system_prompt("You are helpful.")
    .external_mcp_registry(registry)
    .build()
  )

  tool_names = {t.name for t in agent.tools}
  assert "mcp__fs__list" in tool_names
  assert "## Available MCP Servers (external)" in agent.system_prompt
  assert agent.metadata.get("_external_mcp_registry") is registry
