from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mcp.types import CallToolResult, TextContent

from nonoka.ext.mcp.client import MCPCapability, MCPClient, MCPToolError


@pytest.mark.parametrize("attribute", ["input_schema", "inputSchema"])
def test_mcp_capability_reads_current_and_legacy_schema_attributes(attribute):
  schema = {
    "type": "object",
    "properties": {"path": {"type": "string"}},
    "required": ["path"],
  }
  tool = SimpleNamespace(
    name="read_contract",
    description="Read a contract.",
    **{attribute: schema},
  )

  capability = MCPCapability(client=object(), tool=tool)

  assert capability.parameters["required"] == ["path"]
  assert capability.to_json_schema()["function"]["parameters"] == schema


@pytest.mark.asyncio
async def test_mcp_client_reads_sdk_2_snake_case_error_flag():
  client = MCPClient("stdio", command="unused")
  session = AsyncMock()
  session.call_tool.return_value = CallToolResult(
    content=[TextContent(type="text", text="contract unavailable")],
    isError=True,
  )
  client._session = session

  with pytest.raises(MCPToolError, match="contract unavailable"):
    await client.call_tool("read_contract", {"path": "contract.md"})


@pytest.mark.asyncio
async def test_mcp_client_returns_sdk_2_text_content():
  client = MCPClient("stdio", command="unused")
  session = AsyncMock()
  session.call_tool.return_value = CallToolResult(
    content=[TextContent(type="text", text="contract body")],
    isError=False,
  )
  client._session = session

  assert await client.call_tool("read_contract", {}) == "contract body"
