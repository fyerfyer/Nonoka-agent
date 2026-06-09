"""Tests for Gateway session key strategy and message type extensions."""

import pytest
from unittest.mock import AsyncMock

from nonoka.core.agent import Agent
from nonoka.core.runner import Runner
from nonoka.ext.gateway.core import (
  Gateway,
  GatewayMessage,
  GatewayEvent,
  MessageType,
  GatewayEventType,
  _default_session_key_strategy,
  _group_shared_session_key_strategy,
)


# --------------------------------------------------------------------------- #
# Session key strategies
# --------------------------------------------------------------------------- #

def test_default_session_key_strategy():
  """Default strategy should include platform, chat_id, and sender."""
  msg = GatewayMessage(
    message_id="1",
    sender="alice",
    platform="telegram",
    chat_id="group-1",
    content="hello",
  )
  key = _default_session_key_strategy(msg)
  assert key == "telegram:group-1:alice"


def test_group_shared_session_key_strategy():
  """Group strategy should omit sender (shared context per chat)."""
  msg = GatewayMessage(
    message_id="1",
    sender="alice",
    platform="telegram",
    chat_id="group-1",
    content="hello",
  )
  key = _group_shared_session_key_strategy(msg)
  assert key == "telegram:group-1"


def test_custom_session_key_strategy():
  """Gateway should accept a custom session key strategy."""
  runner = Runner()

  def custom_strategy(msg):
    return f"{msg.platform}:{msg.sender}"

  gateway = Gateway(runner, session_key_strategy=custom_strategy)
  msg = GatewayMessage(
    message_id="1",
    sender="bob",
    platform="qq",
    chat_id="chat-1",
    content="hi",
  )
  assert gateway._session_key_strategy(msg) == "qq:bob"


@pytest.mark.asyncio
async def test_gateway_session_isolation_per_sender():
  """Default strategy should create different sessions for different senders."""
  runner = Runner()
  gateway = Gateway(runner)
  gateway.set_default_agent(Agent(model="test", tools=[]))

  # Mock adapter
  mock_adapter = AsyncMock()
  mock_adapter.platform = "tg"
  mock_adapter.send = AsyncMock()
  gateway.register_adapter(mock_adapter)

  # Mock LLM
  from nonoka.core.llm import LLMResponse
  provider = AsyncMock()
  provider.chat = AsyncMock(return_value=LLMResponse(content="ok"))
  runner._llm_cache["test"] = provider
  runner.llm = provider

  msg_alice = GatewayMessage(
    message_id="m1", sender="alice", platform="tg", chat_id="c1", content="hi"
  )
  msg_bob = GatewayMessage(
    message_id="m2", sender="bob", platform="tg", chat_id="c1", content="hi"
  )

  # Process both messages
  await gateway._process_message(msg_alice, gateway._default_agent)
  await gateway._process_message(msg_bob, gateway._default_agent)

  # Different senders should have different session keys
  alice_key = gateway._session_key_strategy(msg_alice)
  bob_key = gateway._session_key_strategy(msg_bob)
  assert alice_key != bob_key

  # Both should have session IDs stored
  assert gateway._session_map.get(alice_key) is not None
  assert gateway._session_map.get(bob_key) is not None
  assert gateway._session_map.get(alice_key) != gateway._session_map.get(bob_key)


@pytest.mark.asyncio
async def test_gateway_group_shared_session():
  """Group strategy should share sessions among senders in the same chat."""
  runner = Runner()
  gateway = Gateway(runner, session_key_strategy=_group_shared_session_key_strategy)
  gateway.set_default_agent(Agent(model="test", tools=[]))

  # Mock adapter
  mock_adapter = AsyncMock()
  mock_adapter.platform = "tg"
  mock_adapter.send = AsyncMock()
  gateway.register_adapter(mock_adapter)

  from nonoka.core.llm import LLMResponse
  provider = AsyncMock()
  provider.chat = AsyncMock(return_value=LLMResponse(content="ok"))
  runner._llm_cache["test"] = provider
  runner.llm = provider

  msg_alice = GatewayMessage(
    message_id="m1", sender="alice", platform="tg", chat_id="c1", content="hi"
  )
  msg_bob = GatewayMessage(
    message_id="m2", sender="bob", platform="tg", chat_id="c1", content="hi"
  )

  await gateway._process_message(msg_alice, gateway._default_agent)
  await gateway._process_message(msg_bob, gateway._default_agent)

  # Same chat should share the same session
  alice_key = gateway._session_key_strategy(msg_alice)
  bob_key = gateway._session_key_strategy(msg_bob)
  assert alice_key == bob_key == "tg:c1"
  assert gateway._session_map.get(alice_key) == gateway._session_map.get(bob_key)


# --------------------------------------------------------------------------- #
# GatewayMessage type extensions
# --------------------------------------------------------------------------- #

def test_gateway_message_defaults_to_text():
  msg = GatewayMessage(
    message_id="1", sender="u1", platform="tg", chat_id="c1", content="hello"
  )
  assert msg.message_type == MessageType.TEXT


def test_gateway_message_image_type():
  msg = GatewayMessage(
    message_id="1",
    sender="u1",
    platform="tg",
    chat_id="c1",
    content="Look at this",
    message_type=MessageType.IMAGE,
    media={"url": "https://example.com/img.jpg", "mime": "image/jpeg"},
  )
  assert msg.message_type == MessageType.IMAGE
  assert msg.media["url"] == "https://example.com/img.jpg"


def test_gateway_message_with_mentions():
  msg = GatewayMessage(
    message_id="1",
    sender="u1",
    platform="tg",
    chat_id="c1",
    content="@alice @bob check this",
    mentions=["alice", "bob"],
  )
  assert msg.mentions == ["alice", "bob"]


# --------------------------------------------------------------------------- #
# GatewayEvent
# --------------------------------------------------------------------------- #

def test_gateway_event_creation():
  event = GatewayEvent(
    event_type=GatewayEventType.MEMBER_JOINED,
    platform="telegram",
    chat_id="group-1",
    sender="alice",
    data={"user_id": "alice", "invited_by": "bob"},
  )
  assert event.event_type == GatewayEventType.MEMBER_JOINED
  assert event.platform == "telegram"
  assert event.data["user_id"] == "alice"
