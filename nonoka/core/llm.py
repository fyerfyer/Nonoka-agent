from enum import Enum
from typing import Any, Protocol, runtime_checkable
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


@runtime_checkable
class LLMProvider(Protocol):
  """
  The framework's LLM calling interface.
  Used to isolate underlying litellm, openai, anthropic, etc. SDKs.
  """
  async def chat(
    self,
    messages: list[LLMMessage],
    tools: list[dict[str, Any]] | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
  ) -> LLMResponse:
    ...

  def count_tokens(self, messages: list[LLMMessage] | str) -> int:
    ...