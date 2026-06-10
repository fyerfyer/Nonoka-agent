import pytest
from unittest.mock import AsyncMock, MagicMock
from collections.abc import AsyncIterator

from nonoka.core.agent import Agent
from nonoka.core.tool import tool
from nonoka.core.context import RunContext
from nonoka.core.runner import Runner

from nonoka.ext.gateway.core import Gateway, GatewayMessage
from nonoka.ext.gateway.limiter import TokenBucketLimiter
from nonoka.ext.gateway.session_map import SessionMap


# --------------------------------------------------------------------------- #
# Mock adapter
# --------------------------------------------------------------------------- #

class MockAdapter:
  """Test-only adapter that records calls."""

  def __init__(self, platform: str = "test"):
    self.platform = platform
    self.sent: list[tuple[str, str, str | None]] = []
    self.streamed: list[tuple[str, list[str], str | None]] = []
    self._on_message: AsyncMock | None = None

  async def start(self, on_message) -> None:
    self._on_message = on_message

  async def send(self, chat_id: str, content: str, reply_to: str | None = None) -> None:
    self.sent.append((chat_id, content, reply_to))

  async def send_stream(self, chat_id: str, stream: AsyncIterator[str], reply_to: str | None = None) -> None:
    chunks = []
    async for chunk in stream:
      chunks.append(chunk)
    self.streamed.append((chat_id, chunks, reply_to))

  async def stop(self) -> None:
    pass

  async def inject(self, msg: GatewayMessage) -> None:
    if self._on_message is not None:
      await self._on_message(msg)


# --------------------------------------------------------------------------- #
# Gateway basic tests
# --------------------------------------------------------------------------- #

def test_gateway_registers_adapter():
  runner = Runner()
  gateway = Gateway(runner)
  adapter = MockAdapter("telegram")
  gateway.register_adapter(adapter)
  assert "telegram" in gateway.adapters


@pytest.mark.asyncio
async def test_gateway_start_starts_adapters():
  runner = Runner()
  gateway = Gateway(runner)
  adapter = MockAdapter("telegram")
  gateway.register_adapter(adapter)
  await gateway.start()
  assert adapter._on_message is not None


@pytest.mark.asyncio
async def test_gateway_send_to_routes_to_adapter():
  runner = Runner()
  gateway = Gateway(runner)
  adapter = MockAdapter("telegram")
  gateway.register_adapter(adapter)
  await gateway.send_to("telegram", "chat-1", "hello")
  assert len(adapter.sent) == 1
  assert adapter.sent[0] == ("chat-1", "hello", None)


@pytest.mark.asyncio
async def test_gateway_send_to_raises_on_unknown_platform():
  runner = Runner()
  gateway = Gateway(runner)
  with pytest.raises(ValueError, match="No adapter registered"):
    await gateway.send_to("unknown", "chat-1", "hello")


# --------------------------------------------------------------------------- #
# Session map
# --------------------------------------------------------------------------- #

def test_session_map_basic():
  sm = SessionMap()
  assert sm.get("key1") is None
  sm.set("key1", "sess-1")
  assert sm.get("key1") == "sess-1"
  sm.delete("key1")
  assert sm.get("key1") is None


# --------------------------------------------------------------------------- #
# Token bucket limiter
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_token_bucket_allows_within_burst():
  limiter = TokenBucketLimiter(default_rate=1, default_burst=3)
  assert await limiter.acquire("user1") is True
  assert await limiter.acquire("user1") is True
  assert await limiter.acquire("user1") is True


@pytest.mark.asyncio
async def test_token_bucket_rejects_when_empty():
  limiter = TokenBucketLimiter(default_rate=1, default_burst=1)
  assert await limiter.acquire("user1") is True
  assert await limiter.acquire("user1") is False


@pytest.mark.asyncio
async def test_token_bucket_per_user_isolation():
  limiter = TokenBucketLimiter(default_rate=1, default_burst=1)
  assert await limiter.acquire("user1") is True
  assert await limiter.acquire("user2") is True  # Different user


# --------------------------------------------------------------------------- #
# Gateway rate limiting
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_gateway_rate_limit_blocks_message():
  runner = Runner()
  limiter = TokenBucketLimiter(default_rate=1, default_burst=0)
  gateway = Gateway(runner, limiter=limiter)
  adapter = MockAdapter("telegram")
  gateway.register_adapter(adapter)
  await gateway.start()

  agent = Agent(model="test", tools=[])
  gateway.set_default_agent(agent)

  msg = GatewayMessage(
    message_id="msg-1",
    sender="alice",
    platform="telegram",
    chat_id="chat-1",
    content="hello",
  )
  await adapter.inject(msg)
  # Should be rate limited — adapter should receive rate limit message
  assert any("Rate limit" in s[1] for s in adapter.sent)


# --------------------------------------------------------------------------- #
# Gateway message processing (with mocked LLM)
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_gateway_processes_message_and_sends_response():
  runner = Runner()
  gateway = Gateway(runner)
  adapter = MockAdapter("telegram")
  gateway.register_adapter(adapter)
  await gateway.start()

  agent = Agent(model="test", tools=[])
  gateway.set_default_agent(agent)

  # Mock LLM so no real call is made
  from nonoka.core.llm import LLMResponse
  fake_llm_response = LLMResponse(content="Hello back", tool_calls=None)

  provider = MagicMock()
  provider.chat = AsyncMock(return_value=fake_llm_response)
  provider.retry_policy = MagicMock(max_retries=0, backoff=0)
  runner._llm_cache["test"] = provider
  runner.llm = provider

  msg = GatewayMessage(
    message_id="msg-1",
    sender="alice",
    platform="telegram",
    chat_id="chat-1",
    content="hello",
  )
  await adapter.inject(msg)

  assert len(adapter.sent) == 1
  assert adapter.sent[0][1] == "Hello back"


# --------------------------------------------------------------------------- #
# Gateway + RunContext reverse channel
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_gateway_auto_binds_to_runner():
  """Gateway.__init__ should automatically set runner.gateway = self."""
  runner = Runner()
  assert runner.gateway is None

  gateway = Gateway(runner)
  assert runner.gateway is gateway


@pytest.mark.asyncio
async def test_gateway_bound_to_session_via_runner():
  """When Gateway is created, _create_session should auto-bind _gateway_ref."""
  runner = Runner()
  gateway = Gateway(runner)
  adapter = MockAdapter("telegram")
  gateway.register_adapter(adapter)

  # Gateway auto-binds to runner — no manual runner.gateway = gateway needed
  agent = Agent(model="test", tools=[])
  session = await runner._create_session(agent, deps=None)

  ctx = RunContext(session)
  assert ctx.gateway is gateway


@pytest.mark.asyncio
async def test_runcontext_gateway_is_none_without_gateway():
  """When no Gateway is attached to runner, ctx.gateway should be None."""
  runner = Runner()
  agent = Agent(model="test", tools=[])
  session = await runner._create_session(agent, deps=None)

  ctx = RunContext(session)
  assert ctx.gateway is None


@pytest.mark.asyncio
async def test_gateway_tool_can_access_ctx_gateway():
  """Tool invoked via Gateway._process_message should be able to use ctx.gateway.send_to."""
  runner = Runner()
  gateway = Gateway(runner)
  adapter = MockAdapter("telegram")
  gateway.register_adapter(adapter)
  await gateway.start()

  # Track whether the tool successfully used ctx.gateway
  tool_called = False
  tool_sent_via_gateway = False

  @tool
  async def notify_admin(ctx: RunContext, message: str) -> str:
    nonlocal tool_called, tool_sent_via_gateway
    tool_called = True
    if ctx.gateway is not None:
      await ctx.gateway.send_to("telegram", "admin-chat", f"ALERT: {message}")
      tool_sent_via_gateway = True
      return "sent"
    return "no gateway"

  agent = Agent(model="test", tools=[notify_admin])
  gateway.set_default_agent(agent)

  # Mock LLM so it calls the tool
  from nonoka.core.llm import LLMResponse
  fake_response_tool = LLMResponse(
    content=None,
    tool_calls=[{
      "id": "tc1",
      "function": {
        "name": "notify_admin",
        "arguments": '{"message": "disk full"}',
      },
    }],
  )
  fake_response_final = LLMResponse(content="Done")

  provider = MagicMock()
  provider.chat = AsyncMock(side_effect=[fake_response_tool, fake_response_final])
  provider.retry_policy = MagicMock(max_retries=0, backoff=0)
  runner._llm_cache["test"] = provider
  runner.llm = provider

  msg = GatewayMessage(
    message_id="msg-1",
    sender="alice",
    platform="telegram",
    chat_id="chat-1",
    content="Alert admin that disk is full",
  )
  await adapter.inject(msg)

  assert tool_called is True, "Tool should have been called"
  assert tool_sent_via_gateway is True, "Tool should have successfully used ctx.gateway"
  # Verify the reverse channel message was sent
  admin_messages = [s for s in adapter.sent if s[0] == "admin-chat"]
  assert len(admin_messages) >= 1
  assert "disk full" in admin_messages[0][1]


@pytest.mark.asyncio
async def test_gateway_weakref_cleared_when_gateway_deleted():
  """When Gateway is garbage-collected, ctx.gateway should return None."""
  runner = Runner()
  gateway = Gateway(runner)
  adapter = MockAdapter("telegram")
  gateway.register_adapter(adapter)

  agent = Agent(model="test", tools=[])
  session = await runner._create_session(agent, deps=None)

  ctx = RunContext(session)
  assert ctx.gateway is gateway

  # Simulate gateway being garbage-collected (e.g., restarted).
  # Must clear runner.gateway first since it holds a strong reference.
  import gc
  runner.gateway = None
  del gateway
  gc.collect()

  # Weak reference should now return None
  assert ctx.gateway is None


# --------------------------------------------------------------------------- #
# Custom agent resolution override
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_gateway_custom_resolve_agent():
  class CustomGateway(Gateway):
    async def _resolve_agent(self, msg):
      if msg.content.startswith("/admin"):
        return self._admin_agent
      return self._default_agent

  runner = Runner()
  gateway = CustomGateway(runner)
  gateway._admin_agent = Agent(model="admin", tools=[])
  gateway._default_agent = Agent(model="default", tools=[])

  resolved = await gateway._resolve_agent(
    GatewayMessage(message_id="1", sender="u", platform="t", chat_id="c", content="/admin help")
  )
  assert resolved.model == "admin"

  resolved = await gateway._resolve_agent(
    GatewayMessage(message_id="2", sender="u", platform="t", chat_id="c", content="hello")
  )
  assert resolved.model == "default"


# --------------------------------------------------------------------------- #
# Multiple gateways on same runner (Bug fix: P3.4)
# --------------------------------------------------------------------------- #

def test_multiple_gateways_do_not_overwrite():
  """Binding multiple Gateways to the same Runner should keep all of them,
  not overwrite the previous one."""
  runner = Runner()
  gateway1 = Gateway(runner)
  gateway2 = Gateway(runner)

  assert runner.gateway is gateway2  # primary is the most recently added
  assert gateway1 in runner._gateways
  assert gateway2 in runner._gateways
  assert len(runner._gateways) == 2


@pytest.mark.asyncio
async def test_gateway_add_gateway_idempotent():
  """Adding the same gateway twice should not duplicate it."""
  runner = Runner()
  gateway = Gateway(runner)

  assert len(runner._gateways) == 1
  runner.add_gateway(gateway)
  assert len(runner._gateways) == 1, "Adding the same gateway twice should be idempotent"


@pytest.mark.asyncio
async def test_runner_gateway_property_backward_compatible():
  """Setting runner.gateway = gw should still work (backward compatibility)."""
  runner = Runner()
  gw1 = Gateway(runner)
  gw2 = Gateway(runner)

  # Setting gateway explicitly should replace all
  runner.gateway = gw1
  assert runner.gateway is gw1
  assert len(runner._gateways) == 1
  assert gw2 not in runner._gateways
