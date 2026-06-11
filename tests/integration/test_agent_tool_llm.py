"""Real LLM integration tests for Agent-as-a-Tool (AgentTool).

These tests use the actual DeepSeek API to verify:
1. Parent Agent can delegate to a sub-agent via AgentTool
2. Sub-agent executes independently and returns results
3. Session isolation works (child does not see parent memory)
4. Depth limiting prevents infinite agent nesting
5. Memory inheritance gives child agent context from parent
6. Cancel propagation stops sub-agent execution
7. Error handling: sub-agent failure is reported to parent

Requirements: OPENAI_API_KEY and OPENAI_BASE_URL in .env
"""

from __future__ import annotations

import os
import pytest
import asyncio

from dotenv import load_dotenv

from nonoka import Agent, Runner, AgentTool, MemoryStrategy
from nonoka.core.tool import tool
from nonoka.core.context import RunContext
from nonoka.core.session import Session

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")
MODEL_NAME = "openai/deepseek-chat" if BASE_URL else "deepseek-chat"


@pytest.fixture
def deepseek_runner():
  """Create a Runner with real DeepSeek provider."""
  if not API_KEY:
    pytest.skip("No OPENAI_API_KEY found in environment.")

  runner = Runner()
  from nonoka.core.llm import LiteLLMProvider
  provider = LiteLLMProvider(
    model=MODEL_NAME,
    api_key=API_KEY,
    base_url=BASE_URL,
  )
  runner._llm_cache[MODEL_NAME] = provider
  runner.llm = provider
  return runner


# --------------------------------------------------------------------------- #
# 1. Basic delegation: parent calls sub-agent via AgentTool
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_agent_tool_basic_delegation(deepseek_runner):
  """A parent agent should be able to delegate a task to a sub-agent
  and receive the sub-agent's result as an observation."""

  # Sub-agent: specialized in summarization
  summarizer = Agent(
    model=MODEL_NAME,
    system_prompt=(
      "You are a summarization expert. When given text, produce a concise "
      "one-sentence summary. Do not ask clarifying questions. "
      "Always provide the summary directly."
    ),
    tools=[],
    max_turns=3,
  )

  # Parent agent: has the summarizer as a tool
  main_agent = Agent(
    model=MODEL_NAME,
    system_prompt=(
      "You are a coordinator. When the user asks for a summary, "
      "use the summarizer tool. After receiving the result, "
      "repeat the summary to the user verbatim."
    ),
    tools=[
      AgentTool(
        agent=summarizer,
        name="summarizer",
        description=(
          "Use this tool when the user asks for a summary of text. "
          "Pass the full text as the 'task' parameter."
        ),
      ),
    ],
    max_turns=5,
  )

  long_text = (
    "Python is a high-level, general-purpose programming language. "
    "Its design philosophy emphasizes code readability with the use of "
    "significant indentation. Python is dynamically typed and garbage-collected. "
    "It supports multiple programming paradigms, including structured, "
    "object-oriented and functional programming."
  )

  result = await deepseek_runner.run_react(
    main_agent,
    prompt=f"Please summarize the following text in one sentence: {long_text}",
    deps=None,
  )

  print(f"\n[Basic delegation] success={result.success}, data={result.data!r}")
  print(f"[Basic delegation] turns={result.session.turn_count}")

  # Should succeed within max_turns
  assert result.success is True, f"Execution failed: {result.error}"
  assert result.session.turn_count <= 5
  # The result should contain a summary (some concise text about Python)
  assert result.data is not None
  assert len(str(result.data)) > 10


# --------------------------------------------------------------------------- #
# 2. Session isolation: child does not see parent memory
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_agent_tool_session_isolation(deepseek_runner):
  """The sub-agent should NOT have access to the parent's conversation history.

  We seed the parent session with a secret fact, then ask the sub-agent
  to recall that fact WITHOUT the parent passing it in the task/context.
  If isolation works, the sub-agent cannot know it.
  """

  # Sub-agent: asked to recall information
  recall_agent = Agent(
    model=MODEL_NAME,
    system_prompt=(
      "You are a recall assistant. Answer based ONLY on the information "
      "provided in the task parameter. If you don't know, say 'I don't know'. "
      "Do not guess."
    ),
    tools=[],
    max_turns=2,
  )

  main_agent = Agent(
    model=MODEL_NAME,
    system_prompt=(
      "You are a test coordinator. When asked about the secret number, "
      "use the recall_helper tool with the question as the task. "
      "IMPORTANT: Do NOT include the secret number in the 'context' parameter. "
      "The recall_helper must figure it out on its own. "
      "After getting the result, report exactly what the tool returned."
    ),
    tools=[
      AgentTool(
        agent=recall_agent,
        name="recall_helper",
        description="Use this to ask questions that require recalling facts.",
      ),
    ],
    max_turns=4,
  )

  # First, tell the parent a secret (this goes into parent memory)
  result1 = await deepseek_runner.run_react(
    main_agent,
    prompt="My secret number is 98765. Remember this.",
    deps=None,
  )
  print(f"\n[Isolation T1] success={result1.success}, data={result1.data!r}")

  # Now ask the parent to delegate a recall question to the sub-agent
  # The sub-agent should NOT know the secret because its session is isolated
  session_id = result1.session.session_id
  result2 = await deepseek_runner.run_react(
    main_agent,
    prompt=(
      "What is my secret number? Use recall_helper to find out. "
      "Do NOT tell recall_helper what the number is."
    ),
    deps=None,
    session_id=session_id,
  )

  print(f"[Isolation T2] success={result2.success}, data={result2.data!r}")
  print(f"[Isolation T2] turns={result2.session.turn_count}")

  assert result2.success is True
  # The sub-agent, not seeing parent memory, should report it doesn't know
  result_text = str(result2.data).lower()
  assert "don't know" in result_text or "not know" in result_text or "unknown" in result_text


# --------------------------------------------------------------------------- #
# 3. Depth limiting prevents infinite nesting
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_agent_tool_depth_limit_blocks(deepseek_runner):
  """When a sub-agent tries to call another AgentTool, depth limiting
  should prevent execution beyond max_depth."""

  # Inner agent: has another AgentTool (self-reference to test depth)
  inner_agent = Agent(
    model=MODEL_NAME,
    system_prompt=(
      "You are an inner agent. If the task asks you to delegate further, "
      "use the nested_agent tool. Otherwise, answer directly."
    ),
    tools=[],  # Will be populated after AgentTool creation
    max_turns=3,
  )

  # Create a self-referencing tool for depth testing
  nested_tool = AgentTool(
    agent=inner_agent,
    name="nested_agent",
    description="Delegate to another agent. Only use if explicitly asked to delegate.",
    max_depth=2,
  )

  # We need to mutate inner_agent.tools after creation because of the circular reference
  # Agent is frozen, so we can't mutate it directly. Instead, create inner_agent with the tool.
  inner_agent = Agent(
    model=MODEL_NAME,
    system_prompt=(
      "You are an inner agent. If the task asks you to delegate further, "
      "use the nested_agent tool. Otherwise, answer directly with a short sentence."
    ),
    tools=[nested_tool],
    max_turns=3,
  )

  # Recreate the tool with the updated agent
  nested_tool = AgentTool(
    agent=inner_agent,
    name="nested_agent",
    description="Delegate to another agent. Only use if explicitly asked to delegate.",
    max_depth=2,
  )

  outer_agent = Agent(
    model=MODEL_NAME,
    system_prompt=(
      "You are an outer agent. Use the nested_agent tool when asked to delegate. "
      "After receiving the result, report it to the user."
    ),
    tools=[nested_tool],
    max_turns=4,
  )

  result = await deepseek_runner.run_react(
    outer_agent,
    prompt=(
      "Please delegate this task to nested_agent: 'Delegate this task again to nested_agent. "
      "The final agent should just say Hello.'"
    ),
    deps=None,
  )

  print(f"\n[Depth limit] success={result.success}, data={result.data!r}")
  print(f"[Depth limit] turns={result.session.turn_count}, error={result.error!r}")

  # With max_depth=2:
  # - Outer (depth 0) calls nested_agent -> inner (depth 1)
  # - Inner (depth 1) tries to call nested_agent -> would be depth 2
  # - But max_depth=2 means depth 2 >= max_depth, so blocked
  # The execution should complete (either success with a partial result,
  # or failure with depth error message)
  assert result.session.turn_count <= 4
  # The result should contain either the final answer or a depth error
  result_text = str(result.data or result.error or "")
  assert "depth" in result_text.lower() or result.success is True


# --------------------------------------------------------------------------- #
# 4. Memory inheritance: child sees parent's recent context
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_agent_tool_memory_inheritance(deepseek_runner):
  """With INHERIT strategy, the sub-agent should see recent parent context."""

  # Sub-agent: answers questions based on provided context
  context_agent = Agent(
    model=MODEL_NAME,
    system_prompt=(
      "You are a context-aware assistant. Answer the user's question "
      "using the conversation context you have been given. Be concise."
    ),
    tools=[],
    max_turns=2,
  )

  main_agent = Agent(
    model=MODEL_NAME,
    system_prompt=(
      "You are a coordinator. When the user asks a follow-up question, "
      "use the context_aware tool to get an answer. Report the tool result verbatim."
    ),
    tools=[
      AgentTool(
        agent=context_agent,
        name="context_aware",
        description="Ask a question with inherited conversation context.",
        memory_strategy=MemoryStrategy.INHERIT,
        inherit_memory_count=3,
      ),
    ],
    max_turns=4,
  )

  # First turn: establish context in parent memory
  result1 = await deepseek_runner.run_react(
    main_agent,
    prompt="We are discussing the planet Mars. It is the fourth planet from the Sun.",
    deps=None,
  )
  print(f"\n[Inherit T1] success={result1.success}")

  # Second turn: ask a follow-up that requires context
  session_id = result1.session.session_id
  result2 = await deepseek_runner.run_react(
    main_agent,
    prompt="What is its position from the Sun? Use context_aware to answer.",
    deps=None,
    session_id=session_id,
  )

  print(f"[Inherit T2] success={result2.success}, data={result2.data!r}")
  print(f"[Inherit T2] turns={result2.session.turn_count}")

  assert result2.success is True
  # The sub-agent, having inherited context, should know Mars is 4th
  assert "4" in str(result2.data) or "fourth" in str(result2.data).lower()


# --------------------------------------------------------------------------- #
# 5. Cancel propagation
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_agent_tool_cancel_propagation(deepseek_runner):
  """When the parent session is cancelled, sub-agent execution should stop."""

  # Sub-agent that takes time (simulated via a slow tool)
  @tool
  async def slow_task(ctx: RunContext) -> str:
    await asyncio.sleep(5.0)
    return "This should not be reached"

  slow_agent = Agent(
    model=MODEL_NAME,
    system_prompt="You are a slow agent. When asked to work, use the slow_task tool.",
    tools=[slow_task],
    max_turns=3,
  )

  main_agent = Agent(
    model=MODEL_NAME,
    system_prompt="You are a coordinator. When asked to run a slow task, use the slow_worker tool.",
    tools=[
      AgentTool(
        agent=slow_agent,
        name="slow_worker",
        description="Run a slow background task.",
      ),
    ],
    max_turns=4,
  )

  # Start execution in background
  task = asyncio.create_task(
    deepseek_runner.run_react(
      main_agent,
      prompt="Run the slow task.",
      deps=None,
    )
  )

  # Wait a moment for the LLM call to start, then cancel
  await asyncio.sleep(2.0)
  task.cancel()

  try:
    result = await task
    print(f"\n[Cancel] result={result}")
  except asyncio.CancelledError:
    print("\n[Cancel] Task was cancelled as expected")
    # This is the expected outcome
    return

  # If we get here, check that the result indicates cancellation
  assert result.error_type == "cancelled" or not result.success


# --------------------------------------------------------------------------- #
# 6. Sub-agent failure handling
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_agent_tool_sub_agent_failure_handled(deepseek_runner):
  """When the sub-agent fails, the error should be reported to the parent
  as an observation, allowing the parent to decide next steps."""

  @tool
  async def always_fail(ctx: RunContext) -> str:
    raise RuntimeError("Intentional failure for testing")

  failing_agent = Agent(
    model=MODEL_NAME,
    system_prompt=(
      "You are a failing agent. When asked to work, use the always_fail tool. "
      "If the tool fails, report the error clearly."
    ),
    tools=[always_fail],
    max_turns=3,
  )

  main_agent = Agent(
    model=MODEL_NAME,
    system_prompt=(
      "You are a resilient coordinator. Use the fragile_worker tool when asked. "
      "If the tool reports an error, tell the user that the operation failed "
      "and include the error message."
    ),
    tools=[
      AgentTool(
        agent=failing_agent,
        name="fragile_worker",
        description="A worker that might fail. Use it when asked to test failure handling.",
      ),
    ],
    max_turns=4,
  )

  result = await deepseek_runner.run_react(
    main_agent,
    prompt="Run fragile_worker. I want to see how errors are handled.",
    deps=None,
  )

  print(f"\n[Failure handling] success={result.success}, data={result.data!r}")
  print(f"[Failure handling] turns={result.session.turn_count}")

  # Parent should receive the error and handle it gracefully
  assert result.data is not None
  result_text = str(result.data).lower()
  # The parent should mention failure or error in its response
  assert "fail" in result_text or "error" in result_text or "could not" in result_text


# --------------------------------------------------------------------------- #
# 7. Complex multi-tool sub-agent
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_agent_tool_complex_sub_agent(deepseek_runner):
  """A sub-agent with multiple tools should be able to use them independently."""

  @tool
  async def get_temperature(ctx: RunContext, city: str) -> dict:
    """Return mock temperature data."""
    temps = {"beijing": 25, "shanghai": 28, "shenzhen": 30}
    return {"city": city, "temperature": temps.get(city.lower(), 20)}

  @tool
  async def get_population(ctx: RunContext, city: str) -> dict:
    """Return mock population data."""
    pops = {"beijing": 21_000_000, "shanghai": 26_000_000, "shenzhen": 17_000_000}
    return {"city": city, "population": pops.get(city.lower(), 0)}

  research_agent = Agent(
    model=MODEL_NAME,
    system_prompt=(
      "You are a research agent. When asked about a city, "
      "use the available tools to gather temperature and population data. "
      "Then summarize both facts in a single sentence."
    ),
    tools=[get_temperature, get_population],
    max_turns=5,
  )

  main_agent = Agent(
    model=MODEL_NAME,
    system_prompt=(
      "You are a coordinator. When the user asks about city statistics, "
      "use the city_researcher tool. Report the result verbatim."
    ),
    tools=[
      AgentTool(
        agent=research_agent,
        name="city_researcher",
        description=(
          "Research a city's statistics. Pass the city name as the 'task' parameter."
        ),
      ),
    ],
    max_turns=5,
  )

  result = await deepseek_runner.run_react(
    main_agent,
    prompt="Tell me about Beijing. Use city_researcher to get the facts.",
    deps=None,
  )

  print(f"\n[Complex sub-agent] success={result.success}, data={result.data!r}")
  print(f"[Complex sub-agent] turns={result.session.turn_count}")

  assert result.success is True
  assert result.data is not None
  result_text = str(result.data).lower()
  # Should contain information about Beijing
  assert "beijing" in result_text
  # Should mention temperature or population (gathered via tools)
  assert any(word in result_text for word in ["temperature", "population", "million", "degrees"])
