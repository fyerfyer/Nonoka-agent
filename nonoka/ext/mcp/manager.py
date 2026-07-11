"""MCP server lifecycle manager for nonoka-agent.

Provides multi-server lifecycle management on top of :class:`MCPClient`:
parallel startup, health checks, exponential-backoff restart, and graceful
shutdown. This is a framework-level primitive so frontends such as nonoka-cli
can configure and reuse it without re-implementing the lifecycle logic.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from nonoka.core.logger import get_logger
from nonoka.core.types import Capability
from nonoka.ext.mcp.client import MCPClient

logger = get_logger("nonoka.mcp.manager")


@dataclass
class MCPServerConfig:
  """Configuration for a single MCP server."""

  transport: Literal["stdio", "sse"]
  command: str | None = None
  args: list[str] = field(default_factory=list)
  env: dict[str, str] | None = None
  cwd: str | None = None
  url: str | None = None


@dataclass
class MCPServerStatus:
  """Runtime status of a managed MCP server."""

  name: str
  status: Literal["connecting", "connected", "restarting", "error", "stopped"]
  transport: Literal["stdio", "sse"]
  tool_count: int = 0
  last_ping: datetime | None = None
  restart_count: int = 0
  error: str | None = None


class MCPManager:
  """Manages the lifecycle of multiple MCP servers and exposes their tools.

  Args:
    health_interval_seconds: How often to ping connected servers.
    max_restart_attempts: Maximum restart attempts after a failure.
    backoff_base_seconds: Base delay for exponential backoff restarts.
  """

  _HEALTH_INTERVAL_SECONDS = 30.0
  _MAX_RESTART_ATTEMPTS = 3
  _BACKOFF_BASE_SECONDS = 2.0

  def __init__(
    self,
    health_interval_seconds: float | None = None,
    max_restart_attempts: int | None = None,
    backoff_base_seconds: float | None = None,
  ):
    self._health_interval_seconds = health_interval_seconds or self._HEALTH_INTERVAL_SECONDS
    self._max_restart_attempts = max_restart_attempts if max_restart_attempts is not None else self._MAX_RESTART_ATTEMPTS
    self._backoff_base_seconds = backoff_base_seconds if backoff_base_seconds is not None else self._BACKOFF_BASE_SECONDS

    self._configs: dict[str, MCPServerConfig] = {}
    self._clients: dict[str, MCPClient] = {}
    self._status: dict[str, MCPServerStatus] = {}
    self._tools: list[tuple[str, Capability]] = []
    self._restart_counts: dict[str, int] = {}
    self._health_task: asyncio.Task[Any] | None = None
    self._stop_event = asyncio.Event()

  # ------------------------------------------------------------------ #
  # Lifecycle
  # ------------------------------------------------------------------ #

  async def start_all(
    self,
    configs: dict[str, MCPServerConfig],
  ) -> list[tuple[str, Capability]]:
    """Start all configured MCP servers in parallel.

    Args:
      configs: Mapping from server name to its configuration.

    Returns:
      A list of ``(server_name, capability)`` pairs from successfully
      started servers.

    Raises:
      MCPRestartExhaustedError: If one or more servers fail to start after
        all attempts. Servers that succeeded are still available.
    """
    self._configs = configs
    self._stop_event.clear()

    results = await asyncio.gather(
      *[self._start_one(name, cfg) for name, cfg in configs.items()],
      return_exceptions=True,
    )

    all_tools: list[tuple[str, Capability]] = []
    failed: list[str] = []
    for name, result in zip(configs.keys(), results):
      if isinstance(result, BaseException):
        logger.error("mcp_server_failed_to_start", name=name, error=str(result))
        failed.append(name)
        self._status[name] = MCPServerStatus(
          name=name,
          status="error",
          transport=configs[name].transport,
          tool_count=0,
          last_ping=None,
          restart_count=self._restart_counts.get(name, 0),
          error=str(result),
        )
      else:
        all_tools.extend((name, cap) for cap in result)

    self._tools = all_tools

    if self._clients:
      self._health_task = asyncio.create_task(self._health_check_loop())

    if failed:
      raise MCPRestartExhaustedError(
        f"Failed to start MCP server(s): {', '.join(failed)}"
      )

    logger.info(
      "mcp_servers_started",
      count=len(self._clients),
      tool_count=len(all_tools),
    )
    return all_tools

  async def start_server(
    self,
    name: str,
    config: MCPServerConfig,
  ) -> list[tuple[str, Capability]]:
    """Start a single MCP server and add it to the managed pool."""
    self._configs[name] = config
    tools = await self._start_one(name, config)
    self._tools.extend((name, cap) for cap in tools)

    if self._health_task is None and self._clients:
      self._health_task = asyncio.create_task(self._health_check_loop())

    logger.info("mcp_server_added", name=name, tool_count=len(tools))
    return [(name, cap) for cap in tools]

  async def _start_one(
    self,
    name: str,
    config: MCPServerConfig,
  ) -> list[Capability]:
    """Start a single MCP server and return its capabilities."""
    self._status[name] = MCPServerStatus(
      name=name,
      status="connecting",
      transport=config.transport,
      tool_count=0,
      last_ping=None,
      restart_count=self._restart_counts.get(name, 0),
      error=None,
    )

    client = MCPClient(
      transport=config.transport,
      command=config.command,
      args=list(config.args),
      env=config.env,
      cwd=config.cwd,
      url=config.url,
    )
    await client.connect()
    tools = await client.get_capabilities()

    self._clients[name] = client
    self._status[name] = MCPServerStatus(
      name=name,
      status="connected",
      transport=config.transport,
      tool_count=len(tools),
      last_ping=datetime.now(),
      restart_count=self._restart_counts.get(name, 0),
      error=None,
    )

    logger.info(
      "mcp_server_connected",
      name=name,
      transport=config.transport,
      tool_count=len(tools),
    )
    return tools

  async def restart(self, name: str) -> list[tuple[str, Capability]]:
    """Restart a single MCP server."""
    if name not in self._configs:
      raise MCPConnectionError(f"MCP server '{name}' is not configured.")

    self._restart_counts[name] = self._restart_counts.get(name, 0) + 1
    restart_count = self._restart_counts[name]

    self._status[name] = MCPServerStatus(
      name=name,
      status="restarting",
      transport=self._configs[name].transport,
      tool_count=0,
      last_ping=None,
      restart_count=restart_count,
      error=None,
    )

    await self._disconnect_one(name)

    config = self._configs[name]
    last_error: BaseException | None = None
    attempts = min(restart_count, self._max_restart_attempts)

    for attempt in range(attempts + 1):
      try:
        tools = await self._start_one(name, config)
        await self._rebuild_tool_list()
        logger.info("mcp_server_restarted", name=name, tool_count=len(tools))
        return [(name, cap) for cap in tools]
      except Exception as exc:  # noqa: BLE001
        last_error = exc
        delay = self._backoff_base_seconds * (2 ** attempt)
        logger.warning(
          "mcp_restart_attempt_failed",
          name=name,
          attempt=attempt + 1,
          delay=delay,
          error=str(exc),
        )
        if attempt < attempts:
          await asyncio.sleep(delay)

    error_msg = f"MCP server '{name}' restart exhausted after {attempts + 1} attempts"
    if last_error is not None:
      error_msg += f": {last_error}"

    self._status[name] = MCPServerStatus(
      name=name,
      status="error",
      transport=config.transport,
      tool_count=0,
      last_ping=None,
      restart_count=restart_count,
      error=str(last_error) if last_error is not None else error_msg,
    )
    await self._rebuild_tool_list()
    raise MCPRestartExhaustedError(error_msg)

  async def stop_all(self) -> None:
    """Gracefully stop all MCP servers and the health-check loop."""
    self._stop_event.set()
    if self._health_task is not None:
      try:
        await asyncio.wait_for(self._health_task, timeout=5.0)
      except asyncio.TimeoutError:
        self._health_task.cancel()
        try:
          await self._health_task
        except asyncio.CancelledError:
          pass
      except asyncio.CancelledError:
        pass
      self._health_task = None

    for name in list(self._clients.keys()):
      await self._disconnect_one(name)

    self._clients.clear()
    self._tools.clear()
    logger.info("mcp_servers_stopped")

  async def _disconnect_one(self, name: str) -> None:
    """Disconnect a single client and mark it stopped."""
    client = self._clients.pop(name, None)
    if client is not None:
      try:
        await client.disconnect()
      except Exception as exc:  # noqa: BLE001
        logger.warning("mcp_disconnect_failed", name=name, error=str(exc))

    status = self._status.get(name)
    if status is not None:
      self._status[name] = MCPServerStatus(
        name=name,
        status="stopped",
        transport=status.transport,
        tool_count=0,
        last_ping=status.last_ping,
        restart_count=status.restart_count,
        error=status.error,
      )

  # ------------------------------------------------------------------ #
  # Health checks
  # ------------------------------------------------------------------ #

  async def _health_check_loop(self) -> None:
    """Periodically ping all connected servers and restart unhealthy ones."""
    while not self._stop_event.is_set():
      try:
        await asyncio.wait_for(
          self._stop_event.wait(),
          timeout=self._health_interval_seconds,
        )
      except asyncio.TimeoutError:
        pass

      if self._stop_event.is_set():
        break

      await self._run_health_check()

  async def _run_health_check(self) -> None:
    """Ping each connected server once and restart any that fail."""
    for name, client in list(self._clients.items()):
      try:
        await client.ping()
        status = self._status[name]
        self._status[name] = MCPServerStatus(
          name=name,
          status="connected",
          transport=status.transport,
          tool_count=status.tool_count,
          last_ping=datetime.now(),
          restart_count=status.restart_count,
          error=None,
        )
      except Exception as exc:  # noqa: BLE001
        logger.warning("mcp_health_check_failed", name=name, error=str(exc))
        status = self._status[name]
        self._status[name] = MCPServerStatus(
          name=name,
          status="error",
          transport=status.transport,
          tool_count=0,
          last_ping=status.last_ping,
          restart_count=status.restart_count,
          error=str(exc),
        )
        try:
          await self.restart(name)
        except MCPRestartExhaustedError:
          logger.error("mcp_auto_restart_exhausted", name=name)

  # ------------------------------------------------------------------ #
  # Tool / status access
  # ------------------------------------------------------------------ #

  def get_status(self, name: str) -> MCPServerStatus:
    """Return the current status for a named server."""
    if name not in self._status:
      raise MCPConnectionError(f"MCP server '{name}' is not known.")
    return self._status[name]

  def list_status(self) -> dict[str, MCPServerStatus]:
    """Return a snapshot of all server statuses."""
    return dict(self._status)

  def get_tools(self) -> list[tuple[str, Capability]]:
    """Return all currently available MCP capabilities with server names."""
    return list(self._tools)

  async def _rebuild_tool_list(self) -> None:
    """Rebuild the merged capability list from all connected clients."""
    tools: list[tuple[str, Capability]] = []
    for name, client in self._clients.items():
      try:
        tools.extend((name, cap) for cap in await client.get_capabilities())
      except Exception as exc:  # noqa: BLE001
        logger.warning("mcp_tool_list_failed", name=name, error=str(exc))
    self._tools = tools


class MCPConnectionError(Exception):
  """Raised when an operation references an unknown MCP server."""
  pass


class MCPRestartExhaustedError(Exception):
  """Raised when an MCP server fails after all restart attempts."""
  pass
