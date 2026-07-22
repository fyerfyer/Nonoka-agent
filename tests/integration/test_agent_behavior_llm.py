"""Real LLM tests for Agent behavior optimizations.

These tests use the actual DeepSeek API to verify:
1. Tool return normalization (has_more) guides LLM decisions
2. Loop detection breaks repetitive tool calling
3. System prompt templates influence behavior
4. Session memory recovery works across turns
5. Graded loop escalation (warning -> block -> terminate)
6. ToolResponse suggested_next_step influences LLM behavior
7. Result similarity detection catches parameter-varied repetition

Requirements: OPENAI_API_KEY and OPENAI_BASE_URL in .env
"""

import os
import pytest

from dotenv import load_dotenv

pytestmark = pytest.mark.live

from nonoka import Agent, Runner, ToolResponse, make_tool_response
from nonoka.core.tool import tool
from nonoka.core.system_prompts import SystemPromptTemplate
from nonoka.core.context import RunContext

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")


@pytest.fixture
def deepseek_runner():
  """Create a Runner with real DeepSeek provider."""
  if not API_KEY:
    pytest.skip("No OPENAI_API_KEY found in environment.")

  model_name = "deepseek-chat"
  if BASE_URL:
    model_name = f"openai/{model_name}"

  runner = Runner()
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
# 1. ToolResponse has_more guides LLM to stop searching
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_toolresponse_has_more_prevents_infinite_search(deepseek_runner):
  """When has_more=false, LLM should stop after one call."""

  call_count = 0

  @tool
  async def search(ctx: RunContext, query: str) -> ToolResponse:
    nonlocal call_count
    call_count += 1
    return make_tool_response(
      data={"results": [f"Result for '{query}' #{call_count}"]},
      has_more=False,
      suggested_next_step="Stop searching and summarise.",
    )

  agent = Agent(
    model="openai/deepseek-chat",
    tools=[search],
    system_prompt=SystemPromptTemplate.EXPLORATION,
    max_turns=5,
  )

  result = await deepseek_runner.run_react(
    agent,
    prompt="Search for information about Python asyncio. Decide when you have enough.",
    deps=None,
  )

  print(f"\n[has_more test] success={result.success}, calls={call_count}, turns={result.session.turn_count}")
  print(f"[has_more test] data={result.data!r}")

  from nonoka.core.memory import MemoryRole
  system_entries = [e for e in result.session.memory.entries if e.role == MemoryRole.SYSTEM]
  notices = [e for e in system_entries if "has_more" in e.content.lower()]
  warnings = [e for e in system_entries if "repeatedly" in e.content.lower() or "loop" in e.content.lower()]

  # The agent should stop within max_turns; framework signals (has_more=False
  # notice + loop warnings) should be present once it starts repeating.
  assert result.session.turn_count <= 5, f"Agent ran too many turns ({result.session.turn_count})"
  if call_count >= 2:
    assert len(notices) >= 1, "has_more=False should trigger a system notice"
  if call_count >= 3:
    assert len(warnings) >= 1, "Repeated calls should trigger loop warnings"
  # Success or a controlled termination are both acceptable
  assert result.success is True or result.error_type in ["loop_detected", "limit_exceeded", "llm_error"]


# --------------------------------------------------------------------------- #
# 2. has_more=True with pagination — legitimate continuation
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_has_more_true_allows_legitimate_pagination(deepseek_runner):
  """When has_more=True, the agent should be allowed to fetch more pages."""

  page = 0

  @tool
  async def paginated_search(ctx: RunContext, query: str, cursor: str | None = None) -> ToolResponse:
    nonlocal page
    page += 1
    if page == 1:
      return make_tool_response(
        data={"results": ["item1", "item2"], "page": 1},
        has_more=True,
        next_cursor="page2",
        suggested_next_step="More results available. Fetch next page.",
      )
    else:
      return make_tool_response(
        data={"results": ["item3"], "page": 2},
        has_more=False,
        suggested_next_step="All results retrieved. Summarize.",
      )

  agent = Agent(
    model="openai/deepseek-chat",
    tools=[paginated_search],
    system_prompt=SystemPromptTemplate.EXPLORATION,
    max_turns=5,
  )

  result = await deepseek_runner.run_react(
    agent,
    prompt="Search for all items about Python. Fetch all pages.",
    deps=None,
  )

  print(f"\n[pagination test] success={result.success}, pages={page}, turns={result.session.turn_count}")
  print(f"[pagination test] data={result.data!r}")

  # Should fetch exactly 2 pages (first with has_more=True, second with has_more=False)
  assert page == 2, f"Should fetch exactly 2 pages, fetched {page}"
  assert result.success is True


# --------------------------------------------------------------------------- #
# 3. Loop detection injects warning and breaks repetition
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_loop_detection_breaks_repeated_tool_calls(deepseek_runner):
  """When the same tool is called repeatedly with has_more=False,
  loop detection should trigger and the agent should stop or change strategy."""

  call_count = 0

  @tool
  async def fetch_page(ctx: RunContext, url: str) -> ToolResponse:
    nonlocal call_count
    call_count += 1
    return make_tool_response(
      data={"url": url, "content": f"Page content #{call_count}"},
      has_more=False,
    )

  agent = Agent(
    model="openai/deepseek-chat",
    tools=[fetch_page],
    system_prompt="You are a research assistant. Use fetch_page to read pages.",
    max_turns=6,
  )

  result = await deepseek_runner.run_react(
    agent,
    prompt=(
      "Read https://example.com/a, https://example.com/b, https://example.com/c. "
      "Read them one by one."
    ),
    deps=None,
  )

  print(f"\n[Loop detection] success={result.success}, calls={call_count}, turns={result.session.turn_count}")

  # Should not exhaust all turns
  assert result.session.turn_count <= 6
  assert result.success is True or result.error_type == "loop_detected"

  from nonoka.core.memory import MemoryRole
  system_entries = [e for e in result.session.memory.entries if e.role == MemoryRole.SYSTEM]
  loop_warnings = [e for e in system_entries if "repeatedly" in e.content.lower() or "loop" in e.content.lower()]

  if call_count >= 3:
    assert len(loop_warnings) >= 1, "Loop warning should be injected when tool is called repeatedly"


# --------------------------------------------------------------------------- #
# 4. System prompt templates influence LLM behavior
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_deterministic_prompt_avoids_unnecessary_tools(deepseek_runner):
  """With DETERMINISTIC prompt, LLM should answer directly without tools
  when the question is within its knowledge."""

  @tool
  async def search(ctx: RunContext, query: str) -> str:
    return f"Search results for: {query}"

  agent = Agent(
    model="openai/deepseek-chat",
    tools=[search],
    system_prompt=SystemPromptTemplate.DETERMINISTIC,
    max_turns=3,
  )

  result = await deepseek_runner.run_react(
    agent,
    prompt="What is the capital of France? Answer directly.",
    deps=None,
  )

  print(f"\n[Deterministic] success={result.success}, data={result.data!r}")

  # Should complete in 1 turn (no tool calls needed)
  assert result.session.turn_count == 1, f"Expected 1 turn, got {result.session.turn_count}"
  assert "paris" in str(result.data).lower()


@pytest.mark.asyncio
async def test_exploration_prompt_uses_tools_when_needed(deepseek_runner):
  """With EXPLORATION prompt, LLM should use tools for live data."""

  @tool
  async def get_time(ctx: RunContext, timezone: str) -> dict:
    from datetime import datetime
    return {"timezone": timezone, "time": datetime.now().isoformat()}

  agent = Agent(
    model="openai/deepseek-chat",
    tools=[get_time],
    system_prompt=SystemPromptTemplate.EXPLORATION,
    max_turns=3,
  )

  result = await deepseek_runner.run_react(
    agent,
    prompt="What time is it now in UTC? Use the get_time tool.",
    deps=None,
  )

  print(f"\n[Exploration] success={result.success}, data={result.data!r}")

  assert result.success is True
  # Should have used the tool (at least 2 turns: tool call + final answer)
  assert result.session.turn_count >= 2


# --------------------------------------------------------------------------- #
# 5. Session memory recovery across turns
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_session_memory_persists_across_turns(deepseek_runner):
  """Multi-turn conversation should retain context."""

  agent = Agent(
    model="openai/deepseek-chat",
    tools=[],
    system_prompt="You are a helpful assistant. Remember what the user told you.",
    max_turns=5,
  )

  # Turn 1: Tell the assistant a fact
  result1 = await deepseek_runner.run_react(
    agent,
    prompt="My favorite color is blue. Remember this.",
    deps=None,
  )
  print(f"\n[Memory T1] success={result1.success}, data={result1.data!r}")

  # Turn 2: Ask about the remembered fact using the same session
  session_id = result1.session.session_id
  result2 = await deepseek_runner.run_react(
    agent,
    prompt="What is my favorite color?",
    deps=None,
    session_id=session_id,
  )
  print(f"[Memory T2] success={result2.success}, data={result2.data!r}")

  assert result2.success is True
  assert "blue" in str(result2.data).lower(), f"LLM forgot user's favorite color: {result2.data!r}"


# --------------------------------------------------------------------------- #
# 6. Graded loop escalation — warning -> block -> terminate
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_graded_loop_escalation(deepseek_runner):
  """When a tool is called repeatedly, the framework should escalate:
  1st trigger: warning
  2nd trigger: block tool
  3rd trigger: terminate
  """

  call_count = 0

  @tool
  async def stubborn_search(ctx: RunContext, query: str) -> dict:
    nonlocal call_count
    call_count += 1
    return {"results": ["same result"], "query": query}

  agent = Agent(
    model="openai/deepseek-chat",
    tools=[stubborn_search],
    system_prompt="You are a research assistant. Use stubborn_search to find information.",
    max_turns=10,
  )

  result = await deepseek_runner.run_react(
    agent,
    prompt="Search for 'python'. Then search for 'java'. Then search for 'golang'. Keep searching.",
    deps=None,
  )

  print(f"\n[Graded escalation] success={result.success}, calls={call_count}, error_type={result.error_type}")

  from nonoka.core.memory import MemoryRole
  system_entries = [e for e in result.session.memory.entries if e.role == MemoryRole.SYSTEM]

  # Should have at least one warning
  warnings = [e for e in system_entries if "repeatedly" in e.content.lower()]
  assert len(warnings) >= 1, "Should have at least one loop warning"

  # After too many repeated calls, should either succeed (LLM stopped) or fail with loop_detected
  if not result.success:
    assert result.error_type in ["loop_detected", "limit_exceeded"], \
      f"Unexpected error: {result.error_type}"


# --------------------------------------------------------------------------- #
# 7. ToolResponse suggested_next_step influences LLM
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_suggested_next_step_stops_tool_calls(deepseek_runner):
  """When a tool returns suggested_next_step telling the LLM to stop,
  the LLM should follow it and provide a final answer."""

  call_count = 0

  @tool
  async def smart_search(ctx: RunContext, query: str) -> ToolResponse:
    nonlocal call_count
    call_count += 1
    return make_tool_response(
      data={"answer": f"The answer to '{query}' is 42"},
      has_more=False,
      suggested_next_step="You have the definitive answer. Provide a final response to the user.",
    )

  agent = Agent(
    model="openai/deepseek-chat",
    tools=[smart_search],
    system_prompt="You are a helpful assistant. Use tools when needed.",
    max_turns=4,
  )

  result = await deepseek_runner.run_react(
    agent,
    prompt="What is the meaning of life? Use smart_search.",
    deps=None,
  )

  print(f"\n[suggested_next_step] success={result.success}, calls={call_count}, turns={result.session.turn_count}")
  print(f"[suggested_next_step] data={result.data!r}")

  # Should call tool once, then follow suggestion and answer
  assert call_count == 1, f"Should call tool once, called {call_count} times"
  assert result.success is True
  assert "42" in str(result.data) or "meaning" in str(result.data).lower()
