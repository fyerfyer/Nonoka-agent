from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


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


# ------------------------------------------------------------------ #
# Default implementation: LiteLLM
# ------------------------------------------------------------------ #

import litellm

from nonoka.core.logger import get_logger


class LiteLLMProvider:
  """
  Litellm-based default LLM gateway.
  Supports 100+ large models, no code changes required to switch freely.
  """

  def __init__(
    self,
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
    **kwargs: Any,
  ):
    self.model = model
    self.api_key = api_key
    self.base_url = base_url
    self.kwargs = kwargs

  async def chat(
    self,
    messages: list[LLMMessage],
    tools: list[dict[str, Any]] | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
  ) -> LLMResponse:
    logger = get_logger(__name__)

    litellm_msgs = [m.model_dump(exclude_none=True) for m in messages]

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

    try:
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
      logger.error(f"LiteLLM completion failed for model {self.model}: {e}")
      raise

  def count_tokens(self, messages: list[LLMMessage] | str) -> int:
    if isinstance(messages, str):
      return litellm.token_counter(model=self.model, text=messages)
    else:
      litellm_msgs = [m.model_dump(exclude_none=True) for m in messages]
      return litellm.token_counter(model=self.model, messages=litellm_msgs)