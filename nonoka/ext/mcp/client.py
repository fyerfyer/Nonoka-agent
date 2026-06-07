from __future__ import annotations

"""
MCP (Model Context Protocol) Client for Nonoka.

Provides seamless integration with MCP servers so that tools exposed via
MCP can be used directly in Nonoka ``Agent`` configurations.

Usage — stdio server::

    from nonoka.ext.mcp import MCPClient

    mcp = MCPClient("stdio", command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", "/path"])
    await mcp.connect()
    tools = await mcp.list_tools()

    agent = Agent(model="gpt-4o", tools=[*local_tools, *mcp.tools])

Usage — SSE server::

    mcp = MCPClient("sse", url="http://localhost:3000/sse")
    await mcp.connect()
    tools = await mcp.list_tools()

    agent = Agent(model="gpt-4o", tools=[*local_tools, *mcp.tools])

Usage — async context manager (recommended)::

    async with MCPClient("stdio", command="python", args=["server.py"]) as mcp:
        tools = await mcp.list_tools()
        agent = Agent(model="gpt-4o", tools=mcp.tools)
        result = await runner.run_react(agent, "List files")
"""

import asyncio
from contextlib import AsyncExitStack
from typing import Any, Protocol, runtime_checkable

from nonoka.core.types import Capability
from nonoka.core.context import RunContext
from nonoka.core.logger import get_logger

logger = get_logger("nonoka.mcp")

# MCP SDK imports
try:
  from mcp import ClientSession
  from mcp.client.stdio import stdio_client, StdioServerParameters
  from mcp.client.sse import sse_client
  from mcp.types import Tool as MCPTool, CallToolResult, TextContent
except ImportError as _exc:  # pragma: no cover
  raise ImportError(
    "The 'mcp' package is required for MCP support. "
    "Install it with: uv add mcp"
  ) from _exc


# --------------------------------------------------------------------------- #
# MCPClient
# --------------------------------------------------------------------------- #

class MCPClient:
  """Lightweight MCP client that bridges MCP servers into Nonoka Agents.

  Supports two transport modes:
  * ``stdio`` — spawn a local subprocess (e.g. npx, python).
  * ``sse`` — connect to a remote SSE endpoint.

  The client exposes discovered tools as ``Capability`` objects so they
  can be passed directly to ``Agent(tools=[...])``.
  """

  def __init__(
    self,
    transport: str,
    *,
    # stdio args
    command: str | None = None,
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
    # sse args
    url: str | None = None,
  ):
    self.transport = transport
    self.command = command
    self.args = args or []
    self.env = env
    self.cwd = cwd
    self.url = url

    self._session: ClientSession | None = None
    self._exit_stack = AsyncExitStack()
    self._tools: list[MCPTool] = []
    self._capability_cache: dict[str, MCPCapability] = {}

  # ------------------------------------------------------------------ #
  # Connection lifecycle
  # ------------------------------------------------------------------ #

  async def connect(self) -> None:
    """Establish connection and initialize the MCP session."""
    if self.transport == "stdio":
      if not self.command:
        raise ValueError("stdio transport requires 'command' parameter")
      await self._connect_stdio()
    elif self.transport == "sse":
      if not self.url:
        raise ValueError("sse transport requires 'url' parameter")
      await self._connect_sse()
    else:
      raise ValueError(f"Unknown transport: {self.transport} (expected 'stdio' or 'sse')")

  async def _connect_stdio(self) -> None:
    params = StdioServerParameters(
      command=self.command,
      args=self.args,
      env=self.env,
      cwd=self.cwd,
    )
    read_stream, write_stream = await self._exit_stack.enter_async_context(
      stdio_client(params)
    )
    self._session = await self._exit_stack.enter_async_context(
      ClientSession(read_stream, write_stream)
    )
    await self._session.initialize()
    logger.info("mcp.connected", transport="stdio", command=self.command)

  async def _connect_sse(self) -> None:
    read_stream, write_stream = await self._exit_stack.enter_async_context(
      sse_client(self.url)
    )
    self._session = await self._exit_stack.enter_async_context(
      ClientSession(read_stream, write_stream)
    )
    await self._session.initialize()
    logger.info("mcp.connected", transport="sse", url=self.url)

  async def disconnect(self) -> None:
    """Close the connection and clean up resources."""
    await self._exit_stack.aclose()
    self._session = None
    self._tools = []
    self._capability_cache = {}
    logger.info("mcp.disconnected")

  async def __aenter__(self) -> MCPClient:
    await self.connect()
    return self

  async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
    await self.disconnect()

  # ------------------------------------------------------------------ #
  # Tool discovery
  # ------------------------------------------------------------------ #

  @property
  def session(self) -> ClientSession:
    if self._session is None:
      raise RuntimeError("MCPClient is not connected. Call connect() first.")
    return self._session

  async def list_tools(self) -> list[MCPTool]:
    """List tools exposed by the MCP server."""
    result = await self.session.list_tools()
    self._tools = result.tools
    logger.info("mcp.tools_discovered", count=len(self._tools))
    return self._tools

  @property
  def tools(self) -> list[Capability]:
    """Return discovered tools as Nonoka ``Capability`` objects.

    Call ``list_tools()`` before accessing this property, or use
    ``get_capabilities()`` which does both.
    """
    return [self._wrap_tool(t) for t in self._tools]

  async def get_capabilities(self) -> list[Capability]:
    """Discover tools and return them as Nonoka capabilities."""
    await self.list_tools()
    return self.tools

  def _wrap_tool(self, tool: MCPTool) -> MCPCapability:
    """Wrap an MCP Tool into a Nonoka Capability."""
    if tool.name in self._capability_cache:
      return self._capability_cache[tool.name]
    cap = MCPCapability(client=self, tool=tool)
    self._capability_cache[tool.name] = cap
    return cap

  # ------------------------------------------------------------------ #
  # Tool invocation
  # ------------------------------------------------------------------ #

  async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
    """Call an MCP tool by name."""
    result: CallToolResult = await self.session.call_tool(name, arguments=arguments)

    if result.isError:
      # Extract error text from content
      error_text = ""
      for item in result.content:
        if isinstance(item, TextContent):
          error_text += item.text
      raise MCPToolError(f"MCP tool '{name}' returned error: {error_text}")

    # Extract text content from result
    texts: list[str] = []
    for item in result.content:
      if isinstance(item, TextContent):
        texts.append(item.text)

    if len(texts) == 1:
      return texts[0]
    return texts


# --------------------------------------------------------------------------- #
# MCPCapability — adapter from MCP Tool to Nonoka Capability
# --------------------------------------------------------------------------- #

class MCPCapability(Capability):
  """Wraps an MCP Tool so it satisfies Nonoka's ``Capability`` Protocol."""

  def __init__(self, client: MCPClient, tool: MCPTool):
    self._client = client
    self._tool = tool

  @property
  def name(self) -> str:
    return self._tool.name

  @property
  def description(self) -> str:
    return self._tool.description or ""

  @property
  def parameters(self) -> dict[str, Any]:
    """Return JSON Schema for the tool's input parameters."""
    return self._tool.inputSchema or {"type": "object", "properties": {}}

  async def invoke(self, ctx: RunContext, arguments: dict[str, Any]) -> Any:
    return await self._client.call_tool(self._tool.name, arguments)

  def to_json_schema(self) -> dict[str, Any]:
    """OpenAI-compatible function schema."""
    return {
      "type": "function",
      "function": {
        "name": self.name,
        "description": self.description,
        "parameters": self.parameters,
      },
    }


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #

class MCPToolError(Exception):
  """Raised when an MCP tool returns an error."""
  pass
