import os
import pytest
from nonoka.core.llm import LLMMessage, LLMMessageRole
from nonoka.core.llm import LiteLLMProvider


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