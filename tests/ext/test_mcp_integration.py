import pytest
import os

from nonoka.ext.mcp import MCPClient


@pytest.mark.asyncio
async def test_mcp_stdio_filesystem_server():
  """Integration test: connect to a real MCP filesystem server via stdio."""
  async with MCPClient(
    "stdio",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
  ) as client:
    # 1. List tools
    tools = await client.list_tools()
    assert len(tools) > 0, "Expected at least one tool from filesystem server"

    tool_names = [t.name for t in tools]
    assert "read_file" in tool_names, f"Expected read_file in tools, got {tool_names}"
    assert "list_directory" in tool_names, f"Expected list_directory in tools, got {tool_names}"

    # 2. Check capability wrappers
    caps = client.tools
    assert len(caps) == len(tools)
    read_file_cap = next((c for c in caps if c.name == "read_file"), None)
    assert read_file_cap is not None
    assert "read" in read_file_cap.description.lower() or read_file_cap.description
    assert "path" in str(read_file_cap.parameters).lower()

    # 3. Call a tool: list_directory
    result = await client.call_tool("list_directory", {"path": "/tmp"})
    assert result is not None
    # Result should be a string or list of strings
    print(f"list_directory result: {result}")

    # 4. Call a tool: read_file (create a test file first)
    test_file = "/tmp/nonoka_mcp_test.txt"
    with open(test_file, "w") as f:
      f.write("Hello from Nonoka MCP test!")

    try:
      result = await client.call_tool("read_file", {"path": test_file})
      assert "Hello from Nonoka MCP test!" in str(result)
      print(f"read_file result: {result}")
    finally:
      os.remove(test_file)


@pytest.mark.asyncio
async def test_mcp_stdio_server_with_agent():
  """Integration test: use MCP tools with a Nonoka Agent."""
  from nonoka import Agent
  from nonoka.core.runner import Runner

  async with MCPClient(
    "stdio",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
  ) as client:
    caps = await client.get_capabilities()
    assert len(caps) > 0

    # Create an agent with MCP tools
    agent = Agent(
      model="deepseek-chat",
      tools=caps,
      system_prompt="You are a helpful assistant with filesystem access.",
      metadata={"source": "mcp_integration_test"},
    )

    # Just verify agent is set up correctly
    assert len(agent.tools) > 0
    assert agent.metadata["source"] == "mcp_integration_test"

    # Check that at least one tool is the read_file tool
    tool_names = [t.name for t in agent.tools]
    assert "read_file" in tool_names
