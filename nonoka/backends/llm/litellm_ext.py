from typing import Any
import logging

from nonoka.core.llm import LLMProvider, LLMMessage, LLMResponse

logger = logging.getLogger(__name__)

try:
  import litellm
  LITELLM_AVAILABLE = True
except ImportError:
  LITELLM_AVAILABLE = False


class LiteLLMProvider(LLMProvider):
  """
  Litellm-based default LLM gateway.
  Supports 100+ large models, no code changes required to switch freely.
  """

  def __init__(
    self,
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
    **kwargs: Any
  ):
    if not LITELLM_AVAILABLE:
      raise ImportError(
        "The litellm library is required. Install it with: pip install litellm"
      )
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
    # Convert framework's LLMMessage to litellm-compatible dict format
    litellm_msgs = [m.model_dump(exclude_none=True) for m in messages]

    # Prepare arguments
    call_kwargs = {
      "model": self.model,
      "messages": litellm_msgs,
      **self.kwargs
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
      # Call underlying model
      response = await litellm.acompletion(**call_kwargs)
      choice = response.choices[0].message

      # Parse tool calls (if any)
      tool_calls = None
      if hasattr(choice, "tool_calls") and choice.tool_calls:
        tool_calls = [
          tc.model_dump() if hasattr(tc, "model_dump") else dict(tc)
          for tc in choice.tool_calls
        ]

      # Record token consumption
      usage_dict = dict(response.usage) if hasattr(response, "usage") and response.usage else {}

      return LLMResponse(
        content=choice.content,
        tool_calls=tool_calls,
        usage=usage_dict
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