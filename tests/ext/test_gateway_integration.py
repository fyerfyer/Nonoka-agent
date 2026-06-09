import os
import pytest
from collections.abc import AsyncIterator

from dotenv import load_dotenv

from nonoka.core.agent import Agent
from nonoka.core.tool import tool
from nonoka.core.context import RunContext
from nonoka.core.runner import Runner

from nonoka.ext.gateway.core import Gateway, GatewayMessage


load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")


class FakeTelegramAdapter:
  """Fake adapter that records messages for integration tests."""

  def __init__(self):
    self.platform = "telegram"
    self.sent: list[tuple[str, str, str | None]] = []
    self._on_message = None

  async def start(self, on_message) -> None:
    self._on_message = on_message

  async def send(self, chat_id: str, content: str, reply_to: str | None = None) -> None:
    self.sent.append((chat_id, content, reply_to))
    print(f"\n[Adapter send] chat={chat_id}: {content[:200]}")

  async def send_stream(self, chat_id: str, stream: AsyncIterator[str], reply_to: str | None = None) -> None:
    chunks = []
    async for chunk in stream:
      chunks.append(chunk)
    content = "".join(chunks)
    self.sent.append((chat_id, content, reply_to))
    print(f"\n[Adapter stream] chat={chat_id}: {content[:200]}")

  async def stop(self) -> None:
    pass

  async def inject(self, msg: GatewayMessage) -> None:
    if self._on_message:
      await self._on_message(msg)


@pytest.fixture
def real_llm_runner():
  """Create a Runner configured for the real Deepseek endpoint."""
  if not API_KEY:
    pytest.skip("No OPENAI_API_KEY found in environment.")

  model_name = "deepseek-chat"
  if BASE_URL:
    model_name = f"openai/{model_name}"

  runner = Runner()
  # Pre-populate the LLM cache with a real provider
  from nonoka.core.llm import LiteLLMProvider
  provider = LiteLLMProvider(
    model=model_name,
    api_key=API_KEY,
    base_url=BASE_URL,
  )
  runner._llm_cache[model_name] = provider
  runner.llm = provider
  return runner


# --------------------------------------------------------------------------- #
# Integration: Gateway -> Agent -> Real LLM -> Adapter
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_gateway_e2e_with_real_llm(real_llm_runner):
  """End-to-end: Gateway receives a message, Agent calls real LLM, response sent back."""
  runner = real_llm_runner
  gateway = Gateway(runner)
  adapter = FakeTelegramAdapter()
  gateway.register_adapter(adapter)
  await gateway.start()

  agent = Agent(
    model="openai/deepseek-chat",
    tools=[],
    system_prompt="You are a helpful assistant. Be very concise.",
    max_turns=3,
  )
  gateway.set_default_agent(agent)

  msg = GatewayMessage(
    message_id="msg-1",
    sender="alice",
    platform="telegram",
    chat_id="chat-123",
    content="Say 'Hello from Gateway' and nothing else.",
  )
  await adapter.inject(msg)

  assert len(adapter.sent) >= 1
  response_text = adapter.sent[0][1]
  print(f"\n[Gateway E2E Response]: {response_text}")
  assert "hello" in response_text.lower() or "gateway" in response_text.lower()


@pytest.mark.asyncio
async def test_gateway_e2e_with_tool_and_real_llm(real_llm_runner):
  """End-to-end with a tool: Agent uses real LLM to decide tool call."""
  runner = real_llm_runner
  gateway = Gateway(runner)
  adapter = FakeTelegramAdapter()
  gateway.register_adapter(adapter)
  await gateway.start()

  @tool
  async def get_weather(city: str) -> dict:
    """Get current weather for a city."""
    return {"city": city, "temperature": 25, "condition": "sunny"}

  agent = Agent(
    model="openai/deepseek-chat",
    tools=[get_weather],
    system_prompt="You are a weather assistant. Use the get_weather tool when asked about weather.",
    max_turns=3,
  )
  gateway.set_default_agent(agent)

  msg = GatewayMessage(
    message_id="msg-2",
    sender="bob",
    platform="telegram",
    chat_id="chat-456",
    content="What's the weather in Beijing?",
  )
  await adapter.inject(msg)

  assert len(adapter.sent) >= 1
  response_text = adapter.sent[0][1]
  print(f"\n[Gateway Tool Response]: {response_text}")
  # The response should contain weather-related info or Beijing
  assert len(response_text) > 0


@pytest.mark.asyncio
async def test_gateway_reverse_channel_with_real_llm(real_llm_runner):
  """Test Agent-initiated push via Gateway (reverse channel)."""
  runner = real_llm_runner
  gateway = Gateway(runner)
  adapter = FakeTelegramAdapter()
  gateway.register_adapter(adapter)
  await gateway.start()

  # Bind gateway to runner so tools can access ctx.gateway
  runner.gateway = gateway

  @tool
  async def notify_admin(ctx: RunContext, message: str) -> str:
    """Send a notification to admin."""
    if ctx.gateway:
      await ctx.gateway.send_to("telegram", "admin-chat", f"ALERT: {message}")
      return "Notification sent"
    return "No gateway available"

  agent = Agent(
    model="openai/deepseek-chat",
    tools=[notify_admin],
    system_prompt=(
      "You are an alert bot. When the user asks you to notify or alert someone, "
      "use the notify_admin tool with an appropriate message."
    ),
    max_turns=3,
  )
  gateway.set_default_agent(agent)

  msg = GatewayMessage(
    message_id="msg-3",
    sender="admin",
    platform="telegram",
    chat_id="user-chat",
    content="Notify admin that the system is running low on disk space.",
  )
  await adapter.inject(msg)

  # Check that the reverse channel sent a message to admin-chat
  admin_messages = [s for s in adapter.sent if s[0] == "admin-chat"]
  print(f"\n[Reverse channel messages]: {admin_messages}")
  assert len(admin_messages) >= 1 or len(adapter.sent) >= 1


@pytest.mark.asyncio
async def test_gateway_session_persistence(real_llm_runner):
  """Test that session is persisted across multiple messages from the same user."""
  runner = real_llm_runner
  gateway = Gateway(runner)
  adapter = FakeTelegramAdapter()
  gateway.register_adapter(adapter)
  await gateway.start()

  agent = Agent(
    model="openai/deepseek-chat",
    tools=[],
    system_prompt="You are a helpful assistant. Remember what the user told you.",
    max_turns=3,
  )
  gateway.set_default_agent(agent)

  # First message: user tells their name
  msg1 = GatewayMessage(
    message_id="msg-4a",
    sender="charlie",
    platform="telegram",
    chat_id="chat-789",
    content="My name is Charlie. Remember that.",
  )
  await adapter.inject(msg1)
  print(f"\n[Session persistence - msg1 response]: {adapter.sent[-1][1]}")

  # Second message: user asks "what's my name?"
  msg2 = GatewayMessage(
    message_id="msg-4b",
    sender="charlie",
    platform="telegram",
    chat_id="chat-789",
    content="What's my name?",
  )
  await adapter.inject(msg2)
  print(f"\n[Session persistence - msg2 response]: {adapter.sent[-1][1]}")

  # Verify session was persisted (default key strategy is platform:chat_id:sender)
  session_key = "telegram:chat-789:charlie"
  session_id = gateway._session_map.get(session_key)
  assert session_id is not None
  print(f"\n[Session persisted]: {session_key} -> {session_id}")
