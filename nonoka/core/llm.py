from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field

from nonoka.core.logger import get_logger
from nonoka.core.types import RetryPolicy


logger = get_logger(__name__)


class LLMMessageRole(str, Enum):
  SYSTEM = "system"
  USER = "user"
  ASSISTANT = "assistant"
  TOOL = "tool"


class LLMMessage(BaseModel):
  """Unified LLM message structure"""
  role: LLMMessageRole | str
  content: str | None = None
  name: str | None = None
  tool_call_id: str | None = None
  tool_calls: list[dict[str, Any]] | None = None


class LLMResponse(BaseModel):
  """Unified LLM response structure"""
  content: str | None = None
  tool_calls: list[dict[str, Any]] | None = None
  usage: dict[str, Any] = Field(default_factory=dict)


class LLMStreamChunk(BaseModel):
  """Single chunk emitted by a streaming LLM response.

  * ``content_delta`` — incremental text since the previous chunk.
  * ``tool_call_deltas`` — incremental tool-call fragments.  Each item is a
    partial dict; callers must accumulate them if they need the complete
    tool-call payload.
  * ``finish_reason`` — present on the final chunk only (e.g. ``"stop"``).
  """
  content_delta: str | None = None
  tool_call_deltas: list[dict[str, Any]] | None = None
  finish_reason: str | None = None


# ------------------------------------------------------------------ #
# Circuit breaker
# ------------------------------------------------------------------ #

class CircuitState(str, Enum):
  CLOSED = "closed"      # Normal operation
  OPEN = "open"          # Failing fast
  HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreaker:
  """Simple in-memory circuit breaker for LLM calls.

  When the failure count reaches *threshold*, the breaker trips to OPEN for
  *recovery_time* seconds.  After that it enters HALF_OPEN; the next call
  decides whether to close (success) or re-open (failure).

  This implementation is intentionally minimal — it only protects a single
  provider instance.  For multi-process deployments, wrap the breaker in a
  shared backend (Redis, etc.).
  """

  def __init__(
    self,
    threshold: int = 5,
    recovery_time: float = 30.0,
  ):
    self.threshold = threshold
    self.recovery_time = recovery_time
    self._state = CircuitState.CLOSED
    self._failures = 0
    self._last_failure_time: float | None = None
    self._lock = asyncio.Lock()

  @property
  def state(self) -> CircuitState:
    return self._state

  async def call(self, coro_factory: Any) -> Any:
    """Execute *coro_factory* if the circuit allows it."""
    async with self._lock:
      if self._state == CircuitState.OPEN:
        if self._last_failure_time is not None:
          elapsed = asyncio.get_event_loop().time() - self._last_failure_time
          if elapsed >= self.recovery_time:
            self._state = CircuitState.HALF_OPEN
            logger.info("circuit_breaker.half_open")
          else:
            raise CircuitBreakerOpen(
              f"Circuit breaker is OPEN ({self.recovery_time - elapsed:.1f}s remaining)"
            )

    try:
      result = await coro_factory()
    except Exception as exc:
      await self._record_failure(exc)
      raise

    await self._record_success()
    return result

  async def _record_failure(self, exc: Exception) -> None:
    async with self._lock:
      self._failures += 1
      self._last_failure_time = asyncio.get_event_loop().time()
      if self._state == CircuitState.HALF_OPEN or self._failures >= self.threshold:
        self._state = CircuitState.OPEN
        logger.warning(
          "circuit_breaker.opened",
          failures=self._failures,
          exc_type=type(exc).__name__,
        )

  async def _record_success(self) -> None:
    async with self._lock:
      if self._state in (CircuitState.OPEN, CircuitState.HALF_OPEN):
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._last_failure_time = None
        logger.info("circuit_breaker.closed")


class CircuitBreakerOpen(Exception):
  """Raised when the circuit breaker is OPEN."""
  pass


class LLMTransientError(Exception):
  """Transient LLM failure — safe to retry."""
  pass


# ------------------------------------------------------------------ #
# Default implementation: LiteLLM
# ------------------------------------------------------------------ #

import litellm


def _infer_litellm_provider_prefix(model: str) -> str | None:
  """Infer a LiteLLM provider prefix from a bare model name.

  LiteLLM expects model identifiers in the form ``provider/model-name``
  unless a recognized provider alias is used. This helper maps common
  model families to their provider prefixes so users can configure
  ``deepseek-chat`` instead of ``deepseek/deepseek-chat``.

  Returns ``None`` when no known provider can be inferred.
  """
  lowered = model.lower()

  if lowered.startswith(("gpt-", "o1", "o3", "text-")):
    return "openai"
  if lowered.startswith(("claude-", "sonnet", "opus", "haiku")):
    return "anthropic"
  if "deepseek" in lowered:
    return "deepseek"
  if lowered.startswith("gemini"):
    return "gemini"
  if lowered.startswith(("llama", "qwen", "mistral", "mixtral", "phi", "codellama")):
    return "ollama"

  return None


class LiteLLMProvider:
  """
  Litellm-based default LLM gateway.
  Supports 100+ large models, no code changes required to switch freely.

  Features:
  * Automatic retry with exponential backoff for transient failures.
  * Optional circuit breaker to fail fast during outages.
  * Synchronous ``chat()`` and streaming ``chat_stream()`` APIs.
  """

  def __init__(
    self,
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
    retry_policy: RetryPolicy | None = None,
    timeout: float | None = None,
    circuit_breaker: CircuitBreaker | None = None,
    **kwargs: Any,
  ):
    # LiteLLM requires a provider prefix (e.g. "deepseek/deepseek-chat")
    # when the model string does not already include one. Infer common
    # providers from the model name so callers can write "deepseek-chat"
    # instead of the more verbose "deepseek/deepseek-chat".
    if "/" not in model:
      prefix = _infer_litellm_provider_prefix(model)
      if prefix:
        model = f"{prefix}/{model}"
      elif base_url:
        # Custom OpenAI-compatible endpoint without a recognizable model
        # name; fall back to the openai/ provider prefix.
        model = f"openai/{model}"
    self.model = model
    self.api_key = api_key
    self.base_url = base_url
    self.retry_policy = retry_policy or RetryPolicy()
    self.timeout = timeout
    self.circuit_breaker = circuit_breaker
    self.kwargs = kwargs

  # ------------------------------------------------------------------ #
  # Public API
  # ------------------------------------------------------------------ #

  async def chat(
    self,
    messages: list[LLMMessage],
    tools: list[dict[str, Any]] | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
  ) -> LLMResponse:
    """Call the LLM with retry, timeout, and circuit-breaker protection."""

    async def _do_chat() -> LLMResponse:
      return await self._chat_once(messages, tools, temperature, max_tokens)

    if self.circuit_breaker is not None:
      return await self.circuit_breaker.call(_do_chat)
    return await _do_chat()

  async def chat_stream(
    self,
    messages: list[LLMMessage],
    tools: list[dict[str, Any]] | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
  ) -> AsyncIterator[LLMStreamChunk]:
    """Stream the LLM response as an async iterator of chunks."""
    litellm_msgs = [m.model_dump(exclude_none=True) for m in messages]
    call_kwargs = self._build_call_kwargs(litellm_msgs, tools, temperature, max_tokens)
    call_kwargs["stream"] = True

    try:
      stream = await litellm.acompletion(**call_kwargs)
    except Exception as e:
      logger.error(f"LiteLLM streaming failed for model {self.model}: {e}")
      raise

    async for part in stream:
      delta = part.choices[0].delta if hasattr(part, "choices") and part.choices else None
      if delta is None:
        continue

      content_delta = getattr(delta, "content", None)
      tool_deltas = None
      raw_tool_calls = getattr(delta, "tool_calls", None)
      if raw_tool_calls:
        tool_deltas = [
          tc.model_dump() if hasattr(tc, "model_dump") else dict(tc)
          for tc in raw_tool_calls
        ]

      finish = None
      if hasattr(part.choices[0], "finish_reason"):
        finish = part.choices[0].finish_reason

      yield LLMStreamChunk(
        content_delta=content_delta,
        tool_call_deltas=tool_deltas,
        finish_reason=finish,
      )

  def count_tokens(self, messages: list[LLMMessage] | str) -> int:
    if isinstance(messages, str):
      return litellm.token_counter(model=self.model, text=messages)
    else:
      litellm_msgs = [m.model_dump(exclude_none=True) for m in messages]
      return litellm.token_counter(model=self.model, messages=litellm_msgs)

  # ------------------------------------------------------------------ #
  # Internals
  # ------------------------------------------------------------------ #

  async def _chat_once(
    self,
    messages: list[LLMMessage],
    tools: list[dict[str, Any]] | None,
    temperature: float | None,
    max_tokens: int | None,
  ) -> LLMResponse:
    """Single non-streaming attempt with retry logic."""
    litellm_msgs = [m.model_dump(exclude_none=True) for m in messages]
    call_kwargs = self._build_call_kwargs(litellm_msgs, tools, temperature, max_tokens)

    max_attempts = max(1, self.retry_policy.max_retries + 1)
    last_exc: Exception | None = None

    for attempt in range(max_attempts):
      try:
        if self.timeout is not None:
          response = await asyncio.wait_for(
            litellm.acompletion(**call_kwargs),
            timeout=self.timeout,
          )
        else:
          response = await litellm.acompletion(**call_kwargs)

        choice = response.choices[0].message
        tool_calls = None
        if hasattr(choice, "tool_calls") and choice.tool_calls:
          tool_calls = [
            tc.model_dump() if hasattr(tc, "model_dump") else dict(tc)
            for tc in choice.tool_calls
          ]

        usage_dict = dict(response.usage) if hasattr(response, "usage") and response.usage else {}

        return LLMResponse(
          content=choice.content,
          tool_calls=tool_calls,
          usage=usage_dict,
        )

      except Exception as e:
        last_exc = e
        if not self._is_retryable(e) or attempt == max_attempts - 1:
          break
        delay = self.retry_policy.backoff * (2 ** attempt)
        logger.warning(
          "llm.retry",
          model=self.model,
          attempt=attempt + 1,
          max_attempts=max_attempts,
          delay=delay,
          error=str(e),
        )
        await asyncio.sleep(delay)

    logger.error(f"LiteLLM completion failed for model {self.model}: {last_exc}")
    if last_exc is None:
      last_exc = RuntimeError("Unknown LLM failure")
    raise last_exc

  def _build_call_kwargs(
    self,
    litellm_msgs: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    temperature: float | None,
    max_tokens: int | None,
  ) -> dict[str, Any]:
    call_kwargs = {
      "model": self.model,
      "messages": litellm_msgs,
      **self.kwargs,
    }
    if self.api_key:
      call_kwargs["api_key"] = self.api_key
    if self.base_url:
      call_kwargs["base_url"] = self.base_url
    if tools:
      call_kwargs["tools"] = tools
    if temperature is not None:
      call_kwargs["temperature"] = temperature
    if max_tokens is not None:
      call_kwargs["max_tokens"] = max_tokens
    return call_kwargs

  @staticmethod
  def _is_retryable(exc: Exception) -> bool:
    """Return True for transient errors that may succeed on retry."""
    retryable_names = {
      "RateLimitError",
      "Timeout",
      "ServiceUnavailableError",
      "APIError",
      "APIConnectionError",
      "BadGatewayError",
      "InternalServerError",
    }
    return type(exc).__name__ in retryable_names


# Backward-compatible alias for external code that imports the exception.
TransientError = LLMTransientError
