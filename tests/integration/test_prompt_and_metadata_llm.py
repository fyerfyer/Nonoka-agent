import pytest
import os

from nonoka import Agent, prompt, PromptTemplate, Runner
from nonoka.core.tool import tool


@pytest.fixture
def runner():
  return Runner()


@pytest.fixture
def deepseek_available():
  """Skip if Deepseek API key is not configured."""
  key = os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
  if not key:
    pytest.skip("Deepseek API key not available")


# --------------------------------------------------------------------------- #
# PromptTemplate + LLM
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_prompt_template_with_real_llm(runner, deepseek_available):
  """Use a Jinja2 prompt template with a real LLM call."""
  tmpl = PromptTemplate.from_string("""
You are a helpful coding assistant.

The user wants to write a {{language}} function that {{task}}.
Please provide a concise implementation.
""")

  system_prompt = tmpl.render_sync(language="python", task="reverses a string")

  agent = Agent(
    model="deepseek-chat",
    system_prompt=system_prompt,
    metadata={"category": "coding", "language": "python"},
    tags=["coding-assistant"],
  )

  result = await runner.run_react(
    agent,
    prompt="Give me the implementation.",
    deps=None,
  )

  print(f"\n[PromptTemplate LLM] success={result.success}, data={result.data!r}")
  assert result.success is True
  assert result.session is not None
  assert result.session.agent.metadata["category"] == "coding"
  assert result.session.agent.metadata["language"] == "python"
  assert "coding-assistant" in result.session.agent.tags
  # LLM should have returned some code
  assert result.data is not None
  assert len(str(result.data)) > 0


@pytest.mark.asyncio
async def test_prompt_decorator_with_real_llm(runner, deepseek_available):
  """Use @prompt decorated function with a real LLM call."""
  @prompt
  async def review_prompt(code: str, language: str = "python") -> str:
    return (
      f"You are a senior {language} engineer. "
      f"Review this code and suggest one improvement:\n```\n{code}\n```"
    )

  system_prompt = await review_prompt.render(
    code="def add(a, b): return a + b",
    language="python",
  )

  agent = Agent(
    model="deepseek-chat",
    system_prompt=system_prompt,
    metadata={"review_type": "code_review"},
    tags=["review"],
  )

  result = await runner.run_react(
    agent,
    prompt="Please review.",
    deps=None,
  )

  print(f"\n[@prompt LLM] success={result.success}, data={result.data!r}")
  assert result.success is True
  assert result.data is not None
  assert result.session.agent.metadata["review_type"] == "code_review"


# --------------------------------------------------------------------------- #
# Agent metadata through execution
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_agent_metadata_preserved_in_session(runner, deepseek_available):
  """Agent metadata should be accessible from the session after execution."""

  @tool
  async def echo(ctx, message: str) -> str:
    return message

  agent = Agent(
    model="deepseek-chat",
    tools=[echo],
    system_prompt="You have an echo tool. Use it when asked.",
    metadata={
      "version": "1.0",
      "team": "platform",
      "cost_center": "ai-infra",
    },
    tags=["production", "echo-test"],
  )

  result = await runner.run_react(
    agent,
    prompt='Use the echo tool to say "hello metadata"',
    deps=None,
  )

  print(f"\n[Metadata E2E] success={result.success}, data={result.data!r}")
  assert result.session is not None
  session_agent = result.session.agent
  assert session_agent.metadata["version"] == "1.0"
  assert session_agent.metadata["team"] == "platform"
  assert "production" in session_agent.tags
  assert "echo-test" in session_agent.tags


@pytest.mark.asyncio
async def test_partial_prompt_template_with_llm(runner, deepseek_available):
  """Partial prompt templates should work end-to-end."""
  base = PromptTemplate.from_string(
    "You are a {{role}}. Answer concisely.\n\nUser: {{question}}"
  )
  expert = base.partial(role="senior engineer")

  agent = Agent(
    model="deepseek-chat",
    system_prompt=expert.render_sync(question="What is Python?"),
    metadata={"template": "partial"},
  )

  result = await runner.run_react(
    agent,
    prompt="",
    deps=None,
  )

  print(f"\n[Partial Template LLM] success={result.success}")
  assert result.success is True
  assert result.data is not None
  assert len(str(result.data)) > 0
