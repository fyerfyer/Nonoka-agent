import os
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from nonoka.core.llm import (
  LLMMessage,
  LLMMessageRole,
  LiteLLMProvider,
  CircuitBreaker,
  CircuitBreakerOpen
)
from nonoka.core.types import RetryPolicy


# --------------------------------------------------------------------------- #
# Retry / timeout / circuit breaker
# --------------------------------------------------------------------------- #

def test_litellm_provider_identifies_retryable_exceptions():
  """Only transient exceptions should be considered retryable."""
  provider = LiteLLMProvider(model="test")

  class FakeRateLimit(Exception):
    pass

  class FakeTimeout(Exception):
    pass

  class FakeAuthError(Exception):
    pass

  FakeRateLimit.__name__ = "RateLimitError"
  FakeTimeout.__name__ = "Timeout"
  FakeAuthError.__name__ = "AuthenticationError"

  assert provider._is_retryable(FakeRateLimit()) is True
  assert provider._is_retryable(FakeTimeout()) is True
  assert provider._is_retryable(FakeAuthError()) is False


@pytest.mark.asyncio
async def test_litellm_provider_retries_transient_error_then_succeeds():
  """Provider should retry on transient errors and eventually succeed."""
  provider = LiteLLMProvider(
    model="test",
    retry_policy=RetryPolicy(max_retries=2, backoff=0.01),
  )

  fake_response = MagicMock()
  fake_response.choices = [MagicMock()]
  fake_response.choices[0].message = MagicMock(content="hi", tool_calls=None)
  fake_response.usage = None

  class RateLimitError(Exception):
    pass

  side_effect = [
    RateLimitError("throttled"),
    RateLimitError("throttled again"),
    fake_response,
  ]

  with patch("nonoka.core.llm.litellm.acompletion", AsyncMock(side_effect=side_effect)) as mock:
    result = await provider.chat([LLMMessage(role="user", content="hello")])

  assert result.content == "hi"
  assert mock.await_count == 3


@pytest.mark.asyncio
async def test_litellm_provider_respects_timeout():
  """Provider should pass timeout through to asyncio.wait_for."""
  provider = LiteLLMProvider(
    model="test",
    timeout=0.05,
    retry_policy=RetryPolicy(max_retries=0, backoff=0.01),
  )

  with patch("nonoka.core.llm.asyncio.wait_for", new_callable=AsyncMock) as mock_wait:
    mock_wait.side_effect = asyncio.TimeoutError("boom")
    with pytest.raises(asyncio.TimeoutError):
      await provider.chat([LLMMessage(role="user", content="hello")])

    assert mock_wait.await_count == 1


@pytest.mark.asyncio
async def test_circuit_breaker_opens_after_threshold():
  """Circuit breaker should trip after *threshold* consecutive failures."""
  cb = CircuitBreaker(threshold=2, recovery_time=60.0)

  async def fail():
    raise RuntimeError("always fails")

  with pytest.raises(RuntimeError):
    await cb.call(fail)
  assert cb.state == "closed"

  with pytest.raises(RuntimeError):
    await cb.call(fail)
  assert cb.state == "open"

  with pytest.raises(CircuitBreakerOpen):
    await cb.call(fail)


@pytest.mark.asyncio
async def test_circuit_breaker_closes_on_success():
  """Circuit breaker should close after a successful call in half-open state."""
  cb = CircuitBreaker(threshold=1, recovery_time=0.0)

  async def fail():
    raise RuntimeError("fail")

  async def succeed():
    return "ok"

  with pytest.raises(RuntimeError):
    await cb.call(fail)
  assert cb.state == "open"

  result = await cb.call(succeed)
  assert result == "ok"
  assert cb.state == "closed"


# --------------------------------------------------------------------------- #
# Streaming interface
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_litellm_stream_yields_content_deltas():
  """chat_stream should yield accumulated content deltas."""
  provider = LiteLLMProvider(model="test")

  class FakeDelta:
    content = "Hello"

  class FakeChoice:
    delta = FakeDelta()
    finish_reason = None

  class FakeChunk:
    choices = [FakeChoice()]

  async def fake_stream():
    yield FakeChunk()

  with patch("nonoka.core.llm.litellm.acompletion", AsyncMock(return_value=fake_stream())):
    chunks = [c async for c in provider.chat_stream([LLMMessage(role="user", content="hi")])]

  assert len(chunks) == 1
  assert chunks[0].content_delta == "Hello"


# --------------------------------------------------------------------------- #
# Integration test with real LLM
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_litellm_provider_real_call():
  """
  Integration test: Actually call the LLM using keys from .env
  To run this: pytest tests/core/test_llm.py -s
  """
  from dotenv import load_dotenv
  load_dotenv()

  api_key = os.getenv("OPENAI_API_KEY")
  base_url = os.getenv("OPENAI_BASE_URL")

  if not api_key:
    pytest.skip("No OPENAI_API_KEY found in environment, skipping real LLM call.")

  model_name = "deepseek-chat"
  if base_url:
    model_name = f"openai/{model_name}"

  provider = LiteLLMProvider(
    model=model_name,
    api_key=api_key,
    base_url=base_url
  )

  messages = [
    LLMMessage(role=LLMMessageRole.SYSTEM, content="You are a helpful test assistant. Be very concise."),
    LLMMessage(role=LLMMessageRole.USER, content="Say 'Hello World' and nothing else.")
  ]

  response = await provider.chat(messages, max_tokens=20)

  assert response is not None
  assert response.content is not None
  print(f"\n[Real LLM Response]: {response.content}")
  assert "hello" in response.content.lower() or "world" in response.content.lower()

  # 测试 Token 计算
  token_count = provider.count_tokens("Say 'Hello World' and nothing else.")
  assert token_count > 0
  print(f"\n[Token Count]: {token_count}")