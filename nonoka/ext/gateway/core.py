from __future__ import annotations

"""
Gateway — Platform access layer for IM platforms.

Gateway is **not** an LLM routing gateway (litellm already does that well).
It is a **platform access gateway** — responsible for standardizing user
requests from QQ, Telegram, Discord, Slack, WeChat, etc. and routing them
to Agents, then pushing Agent outputs back to the original platforms.

Design principles:
1. Gateway does **not** perceive Agent internal logic — it only handles
   request/response standardisation and delivery.
2. Session persistence across platforms — the same user on different
   platforms can have independent sessions (or share, if configured).
3. Agent reuse — one Agent instance can serve multiple Gateway requests
   (Agent is stateless).
4. Adapters are pluggable — each IM platform has its own Adapter, connected
   via a unified protocol.
5. Rate limiting is done at the Gateway layer — protecting downstream LLM
   and tool resources.

Usage::

    gateway = Gateway(runner)
    gateway.register_adapter(TelegramAdapter(token="..."))

    # Bind a default agent (optional — you can also override _on_message)
    gateway.set_default_agent(agent)

    await gateway.start()

Agent active push (reverse channel)::

    @tool
    async def alert_admin(ctx: RunContext, message: str):
        if ctx.gateway:
            await ctx.gateway.send_to("telegram", "admin_group", f"Alert: {message}")
"""

import weakref
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol, TypeVar

from nonoka.core.agent import Agent
from nonoka.core.context import RunContext
from nonoka.core.session import Session
from nonoka.core.types import RunResult
from nonoka.core.llm import LLMStreamChunk
from nonoka.core.runner import Runner

DepsT = TypeVar("DepsT")
ResultT = TypeVar("ResultT")


# --------------------------------------------------------------------------- #
# GatewayMessage — platform-agnostic standard message format
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# Message type enum
# --------------------------------------------------------------------------- #

class MessageType(str):
  """Standard message types supported by Gateway."""
  TEXT = "text"
  IMAGE = "image"
  VOICE = "voice"
  FILE = "file"
  RICH_TEXT = "rich_text"
  SYSTEM = "system"


@dataclass
class GatewayMessage:
  """Platform-agnostic standard message format.

  Supports both plain text and rich media messages.
  For non-text messages, *content* may be a caption/description,
  and *media* carries the actual media payload.
  """

  message_id: str
  sender: str          # Unique user identifier
  platform: str        # "telegram" | "qq" | "discord" | ...
  chat_id: str         # Group / channel / DM identifier
  content: str
  message_type: str = MessageType.TEXT
  reply_to: str | None = None  # ID of the message being replied to
  media: dict | None = None    # Media payload for image/voice/file: {"url": ..., "mime": ...}
  mentions: list[str] = field(default_factory=list)  # List of mentioned user IDs
  raw: dict = field(default_factory=dict)  # Original platform payload
  timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# --------------------------------------------------------------------------- #
# GatewayEvent — system-level events from the platform
# --------------------------------------------------------------------------- #

class GatewayEventType(str):
  """Standard gateway event types."""
  MEMBER_JOINED = "member_joined"
  MEMBER_LEFT = "member_left"
  FRIEND_REQUEST = "friend_request"
  GROUP_INVITE = "group_invite"
  MESSAGE_EDITED = "message_edited"
  MESSAGE_DELETED = "message_deleted"


@dataclass
class GatewayEvent:
  """System event from a platform adapter.

  Distinct from ``GatewayMessage`` — events are not user messages,
  but platform signals (member joins, friend requests, etc.).
  """

  event_type: str
  platform: str
  chat_id: str
  sender: str
  data: dict = field(default_factory=dict)
  timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# --------------------------------------------------------------------------- #
# GatewayAdapter Protocol
# --------------------------------------------------------------------------- #

class GatewayAdapter(Protocol):
  """Platform adapter protocol.

  Each IM platform implements this protocol to:
  * Receive messages from the platform and forward them to the Gateway.
  * Send messages back to the platform.
  * Optionally support streaming delivery.
  """

  @property
  def platform(self) -> str: ...

  async def start(self, on_message: Callable[[GatewayMessage], Awaitable[None]]) -> None:
    """Start the adapter and begin forwarding messages to *on_message*."""
    ...

  async def send(self, chat_id: str, content: str, reply_to: str | None = None) -> None:
    """Send a plain text message to the platform."""
    ...

  async def send_stream(
    self,
    chat_id: str,
    stream: AsyncIterator[str],
    reply_to: str | None = None,
  ) -> None:
    """Send a streaming response to the platform.

    The platform adapter is responsible for either:
    * Accumulating and flushing at appropriate intervals (e.g. Telegram).
    * Sending each chunk immediately (e.g. WebSocket).
    * Or falling back to ``send()`` if streaming is not supported.
    """
    ...

  async def stop(self) -> None:
    """Gracefully stop the adapter."""
    ...


# --------------------------------------------------------------------------- #
# Gateway
# --------------------------------------------------------------------------- #

# Default session key strategies
def _default_session_key_strategy(msg: GatewayMessage) -> str:
  """Private-chat-level isolation: platform + chat + sender."""
  return f"{msg.platform}:{msg.chat_id}:{msg.sender}"


def _group_shared_session_key_strategy(msg: GatewayMessage) -> str:
  """Group-level shared context: platform + chat only."""
  return f"{msg.platform}:{msg.chat_id}"


class Gateway:
  """Platform access entry point.

  Responsibilities:
  * Register and manage platform adapters.
  * Maintain cross-platform session persistence.
  * Rate limiting (via optional Limiter).
  * Execute Agents via Runner and push responses back to the origin platform.
  * Provide ``send_to()`` for Agent-initiated (reverse channel) pushes.

  The default message flow:
  1. Adapter receives a raw platform message.
  2. Adapter converts it to ``GatewayMessage`` and calls ``Gateway._on_message``.
  3. Gateway looks up (or creates) a session for (platform, sender).
  4. Gateway resolves which Agent should handle the message.
  5. Gateway executes the Agent via Runner.
  6. Gateway pushes the response back through the original adapter.
  """

  def __init__(
    self,
    runner: Runner,
    limiter: "Limiter | None" = None,
    session_key_strategy: Callable[[GatewayMessage], str] | None = None,
  ):
    self.runner = runner
    self.adapters: dict[str, GatewayAdapter] = {}
    self._limiter = limiter

    # Session key strategy — configurable for different isolation levels
    self._session_key_strategy = session_key_strategy or _default_session_key_strategy

    # Ensure the runner knows about this gateway so sessions are bound
    # correctly and tools can access ctx.gateway via RunContext.
    runner.gateway = self

    # Cross-platform session mapping: key -> session_id
    from nonoka.ext.gateway.session_map import SessionMap
    self._session_map = SessionMap()

    # Optional default agent — used when no custom handler is provided
    self._default_agent: Agent | None = None

  # ------------------------------------------------------------------ #
  # Agent binding
  # ------------------------------------------------------------------ #

  def set_default_agent(self, agent: Agent) -> None:
    """Bind a default Agent to this Gateway.

    When a message arrives and no custom ``_on_message`` override handles
    it, this Agent will be used.
    """
    self._default_agent = agent

  # ------------------------------------------------------------------ #
  # Adapter management
  # ------------------------------------------------------------------ #

  def register_adapter(self, adapter: GatewayAdapter) -> None:
    """Register a platform adapter."""
    self.adapters[adapter.platform] = adapter

  async def start(self) -> None:
    """Start all registered adapters."""
    for adapter in self.adapters.values():
      await adapter.start(self._on_message)

  async def stop(self) -> None:
    """Stop all registered adapters."""
    for adapter in self.adapters.values():
      await adapter.stop()

  # ------------------------------------------------------------------ #
  # Inbound message handling
  # ------------------------------------------------------------------ #

  async def _on_message(self, msg: GatewayMessage) -> None:
    """Handle an incoming GatewayMessage from an adapter.

    Subclasses can override this to implement custom routing logic.
    The default implementation uses ``_default_agent``.
    """
    # 1. Rate limiting
    if self._limiter is not None:
      limit_key = f"{msg.platform}:{msg.sender}"
      allowed = await self._limiter.acquire(limit_key)
      if not allowed:
        adapter = self.adapters.get(msg.platform)
        if adapter is not None:
          await adapter.send(msg.chat_id, "Rate limit exceeded. Please try again later.")
        return

    # 2. Resolve agent
    agent = await self._resolve_agent(msg)
    if agent is None:
      return

    # 3. Process the message
    await self._process_message(msg, agent)

  async def _resolve_agent(self, msg: GatewayMessage) -> Agent | None:
    """Resolve which Agent should handle *msg*.

    Subclasses can override this for custom routing.
    """
    return self._default_agent

  async def _process_message(
    self,
    msg: GatewayMessage,
    agent: Agent,
  ) -> None:
    """Execute *agent* for *msg* and push the response back."""
    session_key = self._session_key_strategy(msg)
    session_id = self._session_map.get(session_key)

    adapter = self.adapters.get(msg.platform)
    if adapter is None:
      return

    deps = msg  # GatewayMessage is the deps object

    if getattr(agent, "supports_streaming", False):
      # Streaming execution
      stream = self.runner.run_react_stream(
        agent,
        prompt=msg.content,
        deps=deps,
        session_id=session_id,
      )
      await adapter.send_stream(msg.chat_id, self._extract_stream(stream), reply_to=msg.message_id)
    else:
      # Non-streaming execution
      result = await self.runner.run_react(
        agent,
        prompt=msg.content,
        deps=deps,
        session_id=session_id,
      )

      # Update session mapping
      if result.session is not None:
        self._session_map.set(session_key, result.session.session_id)

      # Push response back
      content = str(result.data) if result.data is not None else ""
      if result.error and not result.success:
        content = f"Error: {result.error}"
      await adapter.send(msg.chat_id, content, reply_to=msg.message_id)

  async def _extract_stream(self, stream: AsyncIterator[Any]) -> AsyncIterator[str]:
    """Extract text chunks from a Runner stream for the adapter."""
    async for event in stream:
      if event.type == "content_delta" and event.data.get("content"):
        yield event.data["content"]
      elif event.type == "final":
        # Final event contains the complete result — no extra yield needed
        pass

  # ------------------------------------------------------------------ #
  # Outbound / reverse channel — Agent-initiated push
  # ------------------------------------------------------------------ #

  async def send_to(self, platform: str, chat_id: str, content: str) -> None:
    """Push a message to a specific platform/chat.

    This is the "reverse channel" that allows Agents (via tools) to
    proactively send messages, not just respond to incoming ones.
    """
    adapter = self.adapters.get(platform)
    if adapter is None:
      raise ValueError(f"No adapter registered for platform: {platform}")
    await adapter.send(chat_id, content)

  # ------------------------------------------------------------------ #
  # Session access for RunContext
  # ------------------------------------------------------------------ #

  def bind_session(self, session: Session) -> None:
    """Bind this Gateway to a Session so tools can access it via RunContext.

    This is called internally by the Runner when a session is created
    for a Gateway request.
    """
    # Store a weak reference on the session so RunContext can access it
    object.__setattr__(session, "_gateway_ref", weakref.ref(self))
