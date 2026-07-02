"""Tests for ReActAgent loop detection logic (mock LLM)."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from nonoka.core.agent import Agent
from nonoka.core.tool import tool
from nonoka.core.paradigm import ReActAgent
from nonoka.core.memory import WorkingMemory, MemoryRole
from nonoka.core.session import Session
from nonoka.core.llm import LLMMessage, LLMResponse


@pytest.fixture
def mock_runner():
  """Create a Runner with a mocked LLM provider."""
  from nonoka.core.runner import Runner
  runner = Runner(checkpoint="memory")
  runner.llm = MagicMock()
  runner.checkpoint_store = MagicMock()
  runner.checkpoint_store.save_session = AsyncMock()
  runner.checkpoint_store.save_step_status = AsyncMock()
  runner.checkpoint_store.save_step_result = AsyncMock()
  runner.checkpoint_store.save_step_error = AsyncMock()
  runner.hooks = MagicMock()
  runner.hooks.emit_session_start = AsyncMock()
  runner.hooks.emit_session_end = AsyncMock()
  runner.hooks.emit_llm_request = AsyncMock()
  runner.hooks.emit_llm_response = AsyncMock()
  runner.hooks.emit_tool_start = AsyncMock()
  runner.hooks.emit_tool_start_intercept = AsyncMock(side_effect=lambda ctx, name, args: args)
  runner.hooks.emit_tool_end = AsyncMock()
  return runner


# --------------------------------------------------------------------------- #
# Helper: build LLM response that calls a single tool
# --------------------------------------------------------------------------- #

def _tool_call(tool_name: str, args: dict, call_id: str = "tc") -> dict:
  import json
  return {
    "id": call_id,
    "function": {
      "name": tool_name,
      "arguments": json.dumps(args),
    },
  }


# --------------------------------------------------------------------------- #
# Loop detection: basic consecutive tool
# --------------------------------------------------------------------------- #

from nonoka.core.tool_response import ToolResponse

@tool
async def search_tool(ctx, query: str) -> ToolResponse:
  return ToolResponse(
    data={"results": [f"result for {query}"]},
    has_more=True,
  )


@pytest.mark.asyncio
async def test_loop_detection_injects_system_warning(mock_runner):
  """When the same tool is called >= max_repeated_tool_calls, a system
  warning should be injected into memory."""
  agent = Agent(
    model="test",
    tools=[search_tool],
    max_turns=5,
    system_prompt="You are a test agent.",
  )

  session = Session(session_id="loop-test", agent=agent, deps=None)
  session.memory = WorkingMemory(session_id="loop-test", memory_backend=None)

  # Simulate LLM calling search_tool 3 times with similar queries
  responses = [
    LLMResponse(
      content=None,
      tool_calls=[_tool_call("search_tool", {"query": "weather"}, f"tc{i}")],
    )
    for i in range(3)
  ]
  responses.append(LLMResponse(content="Done searching."))

  mock_runner.llm.chat = AsyncMock(side_effect=responses)

  paradigm = ReActAgent(max_repeated_tool_calls=3)
  result = await paradigm.run(session, mock_runner, prompt="Search for weather")

  # Check that a SYSTEM warning was injected after the 3rd repeated call
  system_entries = [e for e in session.memory.entries if e.role == MemoryRole.SYSTEM]
  assert len(system_entries) >= 1, "Loop warning should have been injected"
  assert "stop" in system_entries[-1].content.lower()


@pytest.mark.asyncio
async def test_loop_detection_does_not_trigger_prematurely(mock_runner):
  """Loop warning should NOT be injected before the threshold."""
  agent = Agent(
    model="test",
    tools=[search_tool],
    max_turns=5,
  )

  session = Session(session_id="no-loop-test", agent=agent, deps=None)
  session.memory = WorkingMemory(session_id="no-loop-test", memory_backend=None)

  responses = [
    LLMResponse(
      content=None,
      tool_calls=[_tool_call("search_tool", {"query": "weather"}, f"tc{i}")],
    )
    for i in range(2)
  ]
  responses.append(LLMResponse(content="Done."))

  mock_runner.llm.chat = AsyncMock(side_effect=responses)

  paradigm = ReActAgent(max_repeated_tool_calls=3)
  result = await paradigm.run(session, mock_runner, prompt="Search")

  loop_warnings = [
    e for e in session.memory.entries
    if e.role == MemoryRole.SYSTEM and ("loop" in e.content.lower() or "repeatedly" in e.content.lower())
  ]
  assert len(loop_warnings) == 0, "Should not trigger loop warning before threshold"


@pytest.mark.asyncio
async def test_loop_detection_resets_on_different_tool(mock_runner):
  """Calling a different tool should reset the consecutive counter."""

  @tool
  async def other_tool(ctx, data: str) -> str:
    return f"other: {data}"

  agent = Agent(
    model="test",
    tools=[search_tool, other_tool],
    max_turns=5,
  )

  session = Session(session_id="reset-test", agent=agent, deps=None)
  session.memory = WorkingMemory(session_id="reset-test", memory_backend=None)

  responses = [
    LLMResponse(content=None, tool_calls=[_tool_call("search_tool", {"query": "a"}, "tc1")]),
    LLMResponse(content=None, tool_calls=[_tool_call("search_tool", {"query": "b"}, "tc2")]),
    LLMResponse(content=None, tool_calls=[_tool_call("other_tool", {"data": "c"}, "tc3")]),
    LLMResponse(content=None, tool_calls=[_tool_call("search_tool", {"query": "d"}, "tc4")]),
    LLMResponse(content=None, tool_calls=[_tool_call("search_tool", {"query": "e"}, "tc5")]),
    LLMResponse(content="Done."),
  ]

  mock_runner.llm.chat = AsyncMock(side_effect=responses)

  paradigm = ReActAgent(max_repeated_tool_calls=3)
  result = await paradigm.run(session, mock_runner, prompt="Test")

  loop_warnings = [
    e for e in session.memory.entries
    if e.role == MemoryRole.SYSTEM and ("loop" in e.content.lower() or "repeatedly" in e.content.lower())
  ]
  assert len(loop_warnings) == 0, "Different tool should reset consecutive counter"


# --------------------------------------------------------------------------- #
# has_more smart exemption
# --------------------------------------------------------------------------- #

from nonoka.core.tool_response import ToolResponse

@tool
async def paginated_search(ctx, query: str) -> ToolResponse:
  """Returns has_more=True to test exemption logic."""
  return ToolResponse(
    data={"results": [f"page for {query}"]},
    has_more=True,
  )


@pytest.mark.asyncio
async def test_loop_detection_exempts_has_more_true(mock_runner):
  """When tool returns has_more=True, consecutive calls should be exempted
  (counted at half weight) so legitimate pagination doesn't trigger loop."""
  agent = Agent(
    model="test",
    tools=[paginated_search],
    max_turns=6,
  )

  session = Session(session_id="has-more-test", agent=agent, deps=None)
  session.memory = WorkingMemory(session_id="has-more-test", memory_backend=None)

  # Call 4 times — with normal weight this would trigger at threshold=3
  # With half weight for has_more=True, effective count is ~2.5
  responses = [
    LLMResponse(
      content=None,
      tool_calls=[_tool_call("paginated_search", {"query": f"q{i}"}, f"tc{i}")],
    )
    for i in range(4)
  ]
  responses.append(LLMResponse(content="Done."))

  mock_runner.llm.chat = AsyncMock(side_effect=responses)

  paradigm = ReActAgent(max_repeated_tool_calls=3)
  result = await paradigm.run(session, mock_runner, prompt="Search")

  loop_warnings = [
    e for e in session.memory.entries
    if e.role == MemoryRole.SYSTEM and ("loop" in e.content.lower() or "repeatedly" in e.content.lower())
  ]
  assert len(loop_warnings) == 0, "has_more=True should exempt from loop detection"


@tool
async def no_more_search(ctx, query: str) -> ToolResponse:
  """Returns has_more=False to test accelerated detection."""
  return ToolResponse(
    data={"results": [f"final for {query}"]},
    has_more=False,
  )


@pytest.mark.asyncio
async def test_loop_detection_accelerates_when_has_more_false(mock_runner):
  """When tool returns has_more=False but is still called, threshold should
  drop, causing earlier loop detection."""
  agent = Agent(
    model="test",
    tools=[no_more_search],
    max_turns=5,
  )

  session = Session(session_id="no-more-test", agent=agent, deps=None)
  session.memory = WorkingMemory(session_id="no-more-test", memory_backend=None)

  # With threshold=3 and has_more=False, effective threshold drops to 2
  responses = [
    LLMResponse(
      content=None,
      tool_calls=[_tool_call("no_more_search", {"query": f"q{i}"}, f"tc{i}")],
    )
    for i in range(3)
  ]
  responses.append(LLMResponse(content="Done."))

  mock_runner.llm.chat = AsyncMock(side_effect=responses)

  paradigm = ReActAgent(max_repeated_tool_calls=3)
  result = await paradigm.run(session, mock_runner, prompt="Search")

  loop_warnings = [
    e for e in session.memory.entries
    if e.role == MemoryRole.SYSTEM and ("loop" in e.content.lower() or "repeatedly" in e.content.lower())
  ]
  assert len(loop_warnings) >= 1, "has_more=False should accelerate loop detection"


# --------------------------------------------------------------------------- #
# Short-cycle detection (A→B→A→B)
# --------------------------------------------------------------------------- #

@tool
async def tool_a(ctx, data: str) -> str:
  return f"A: {data}"


@tool
async def tool_b(ctx, data: str) -> str:
  return f"B: {data}"


@pytest.mark.asyncio
async def test_loop_detection_catches_alternating_pattern(mock_runner):
  """A→B→A→B pattern should be detected as a loop."""
  agent = Agent(
    model="test",
    tools=[tool_a, tool_b],
    max_turns=6,
  )

  session = Session(session_id="alt-test", agent=agent, deps=None)
  session.memory = WorkingMemory(session_id="alt-test", memory_backend=None)

  responses = [
    LLMResponse(content=None, tool_calls=[_tool_call("tool_a", {"data": "1"}, "tc1")]),
    LLMResponse(content=None, tool_calls=[_tool_call("tool_b", {"data": "2"}, "tc2")]),
    LLMResponse(content=None, tool_calls=[_tool_call("tool_a", {"data": "3"}, "tc3")]),
    LLMResponse(content=None, tool_calls=[_tool_call("tool_b", {"data": "4"}, "tc4")]),
    LLMResponse(content="Done."),
  ]

  mock_runner.llm.chat = AsyncMock(side_effect=responses)

  paradigm = ReActAgent()
  result = await paradigm.run(session, mock_runner, prompt="Test")

  loop_warnings = [
    e for e in session.memory.entries
    if e.role == MemoryRole.SYSTEM and ("loop" in e.content.lower() or "repeatedly" in e.content.lower())
  ]
  assert len(loop_warnings) >= 1, "A→B→A→B pattern should trigger loop detection"


# --------------------------------------------------------------------------- #
# Result similarity detection
# --------------------------------------------------------------------------- #

@tool
async def similar_result_tool(ctx, query: str) -> ToolResponse:
  """Returns nearly identical results regardless of query."""
  return ToolResponse(
    data={"results": ["same result"], "query_used": query},
    has_more=False,
  )


@pytest.mark.asyncio
async def test_loop_detection_by_result_similarity(mock_runner):
  """When a tool returns substantively identical results across calls with
  different arguments, it should be detected as a loop."""
  agent = Agent(
    model="test",
    tools=[similar_result_tool],
    max_turns=6,
  )

  session = Session(session_id="sim-test", agent=agent, deps=None)
  session.memory = WorkingMemory(session_id="sim-test", memory_backend=None)

  # 3 calls with different queries but same output
  responses = [
    LLMResponse(
      content=None,
      tool_calls=[_tool_call("similar_result_tool", {"query": f"q{i}"}, f"tc{i}")],
    )
    for i in range(3)
  ]
  responses.append(LLMResponse(content="Done."))

  mock_runner.llm.chat = AsyncMock(side_effect=responses)

  paradigm = ReActAgent()
  result = await paradigm.run(session, mock_runner, prompt="Test")

  loop_warnings = [
    e for e in session.memory.entries
    if e.role == MemoryRole.SYSTEM and ("loop" in e.content.lower() or "repeatedly" in e.content.lower())
  ]
  assert len(loop_warnings) >= 1, "Similar results should trigger loop detection"


# --------------------------------------------------------------------------- #
# Graded response escalation
# --------------------------------------------------------------------------- #

@tool
async def stubborn_tool(ctx, data: str) -> str:
  return "stubborn result"


@pytest.mark.asyncio
async def test_loop_detection_escalation_blocks_tool(mock_runner):
  """On 2nd loop trigger, the tool should be blocked from further calls."""
  agent = Agent(
    model="test",
    tools=[stubborn_tool],
    max_turns=8,
  )

  session = Session(session_id="escalation-test", agent=agent, deps=None)
  session.memory = WorkingMemory(session_id="escalation-test", memory_backend=None)

  # 6 calls to trigger escalation twice
  responses = [
    LLMResponse(
      content=None,
      tool_calls=[_tool_call("stubborn_tool", {"data": f"d{i}"}, f"tc{i}")],
    )
    for i in range(6)
  ]
  responses.append(LLMResponse(content="Done."))

  mock_runner.llm.chat = AsyncMock(side_effect=responses)

  paradigm = ReActAgent(max_repeated_tool_calls=3)
  result = await paradigm.run(session, mock_runner, prompt="Test")

  # After 2nd trigger, tool should be blocked
  assert "stubborn_tool" in getattr(session, "_blocked_tools", set()), \
    "Tool should be blocked after 2nd loop trigger"


@pytest.mark.asyncio
async def test_loop_detection_termination_on_third_trigger(mock_runner):
  """On 3rd loop trigger, execution should terminate with loop_detected error."""
  agent = Agent(
    model="test",
    tools=[stubborn_tool],
    max_turns=10,
  )

  session = Session(session_id="term-test", agent=agent, deps=None)
  session.memory = WorkingMemory(session_id="term-test", memory_backend=None)

  # 9 calls to trigger escalation three times
  responses = [
    LLMResponse(
      content=None,
      tool_calls=[_tool_call("stubborn_tool", {"data": f"d{i}"}, f"tc{i}")],
    )
    for i in range(9)
  ]
  responses.append(LLMResponse(content="Done."))

  mock_runner.llm.chat = AsyncMock(side_effect=responses)

  paradigm = ReActAgent(max_repeated_tool_calls=3)
  result = await paradigm.run(session, mock_runner, prompt="Test")

  assert result.success is False
  assert result.error_type == "loop_detected"


# --------------------------------------------------------------------------- #
# ToolResponse suggested_next_step propagation
# --------------------------------------------------------------------------- #

@tool
async def guided_tool(ctx, query: str) -> dict:
  return {
    "result": {"answer": 42},
    "has_more": False,
    "suggested_next_step": "Use this answer to provide a final response.",
  }


@pytest.mark.asyncio
async def test_tool_response_suggested_next_step_propagated(mock_runner):
  """When a tool returns suggested_next_step, it should be injected into
  memory as a SYSTEM message."""
  agent = Agent(
    model="test",
    tools=[guided_tool],
    max_turns=3,
  )

  session = Session(session_id="guided-test", agent=agent, deps=None)
  session.memory = WorkingMemory(session_id="guided-test", memory_backend=None)

  responses = [
    LLMResponse(
      content=None,
      tool_calls=[_tool_call("guided_tool", {"query": "test"}, "tc1")],
    ),
    LLMResponse(content="The answer is 42."),
  ]

  mock_runner.llm.chat = AsyncMock(side_effect=responses)

  paradigm = ReActAgent()
  result = await paradigm.run(session, mock_runner, prompt="Ask guided_tool")

  guidance_entries = [
    e for e in session.memory.entries
    if e.role == MemoryRole.SYSTEM and "Tool guidance" in e.content
  ]
  assert len(guidance_entries) >= 1, "suggested_next_step should be propagated"
  assert "final response" in guidance_entries[0].content


@pytest.mark.asyncio
async def test_tool_response_has_more_false_system_notice(mock_runner):
  """When a tool returns has_more=False, a SYSTEM notice should be injected."""
  agent = Agent(
    model="test",
    tools=[guided_tool],
    max_turns=3,
  )

  session = Session(session_id="has-more-false-test", agent=agent, deps=None)
  session.memory = WorkingMemory(session_id="has-more-false-test", memory_backend=None)

  responses = [
    LLMResponse(
      content=None,
      tool_calls=[_tool_call("guided_tool", {"query": "test"}, "tc1")],
    ),
    LLMResponse(content="Done."),
  ]

  mock_runner.llm.chat = AsyncMock(side_effect=responses)

  paradigm = ReActAgent()
  result = await paradigm.run(session, mock_runner, prompt="Test")

  notice_entries = [
    e for e in session.memory.entries
    if e.role == MemoryRole.SYSTEM and "has_more" in e.content
  ]
  assert len(notice_entries) >= 1, "has_more=false should trigger a system notice"


# --------------------------------------------------------------------------- #
# Tool return value normalization
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_tool_invoke_normalizes_plain_return_value():
  """Tool.invoke should wrap plain values in the standard response shape."""

  @tool
  async def plain_tool(ctx, value: str) -> str:
    return f"result: {value}"

  from nonoka.core.context import RunContext
  from nonoka.core.session import Session

  agent = Agent(model="test", tools=[plain_tool])
  session = Session(session_id="norm-test", agent=agent)
  ctx = RunContext(session)

  result = await plain_tool.invoke(ctx, {"value": "hello"})
  assert isinstance(result, dict)
  assert result["result"] == "result: hello"
  assert result["has_more"] is False


@pytest.mark.asyncio
async def test_tool_invoke_expands_toolresponse():
  """Tool.invoke should expand a ToolResponse into its dict form."""
  from nonoka.core.tool_response import ToolResponse

  @tool
  async def rich_tool(ctx, query: str) -> ToolResponse:
    return ToolResponse(
      data={"items": [1, 2]},
      has_more=True,
      next_cursor="page2",
    )

  from nonoka.core.context import RunContext

  agent = Agent(model="test", tools=[rich_tool])
  session = Session(session_id="rich-test", agent=agent)
  ctx = RunContext(session)

  result = await rich_tool.invoke(ctx, {"query": "test"})
  assert isinstance(result, dict)
  assert result["result"] == {"items": [1, 2]}
  assert result["has_more"] is True
  assert result["next_cursor"] == "page2"


# --------------------------------------------------------------------------- #
# Repair-attempt exemption (Bug fix: P1.3)
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_loop_detection_exempts_repair_attempt(mock_runner):
  """When a tool fails first and then succeeds with corrected arguments,
  it should NOT be treated as a loop (repair-attempt exemption)."""

  from nonoka.core.errors import LogicError

  @tool
  async def strict_divide(ctx, numerator: int, denominator: int) -> float:
    if denominator == 0:
      raise LogicError("Cannot divide by zero")
    return numerator / denominator

  agent = Agent(
    model="test",
    tools=[strict_divide],
    max_turns=5,
  )

  session = Session(session_id="repair-test", agent=agent, deps=None)
  session.memory = WorkingMemory(session_id="repair-test", memory_backend=None)

  # Turn 1: wrong argument (denominator=0) → fails (LogicError → REPORT)
  # Turn 2: corrected argument (denominator=2) → succeeds
  responses = [
    LLMResponse(
      content=None,
      tool_calls=[_tool_call("strict_divide", {"numerator": 10, "denominator": 0}, "tc1")],
    ),
    LLMResponse(
      content=None,
      tool_calls=[_tool_call("strict_divide", {"numerator": 10, "denominator": 2}, "tc2")],
    ),
    LLMResponse(content="The result is 5.0."),
  ]

  mock_runner.llm.chat = AsyncMock(side_effect=responses)

  paradigm = ReActAgent(max_repeated_tool_calls=2)
  result = await paradigm.run(session, mock_runner, prompt="Divide 10 by 2")

  assert result.success is True
  loop_warnings = [
    e for e in session.memory.entries
    if e.role == MemoryRole.SYSTEM and ("loop" in e.content.lower() or "repeatedly" in e.content.lower())
  ]
  assert len(loop_warnings) == 0, "Repair attempt (fail then succeed with different args) should not trigger loop detection"


@pytest.mark.asyncio
async def test_loop_detection_exempts_self_healing_tool(mock_runner):
  """A self-healing tool that fails once then succeeds should not trigger
  loop detection — this is the exact scenario from the bug report."""

  from nonoka.core.errors import LogicError

  call_count = 0

  @tool
  async def self_healing_tool(ctx, mode: str) -> str:
    nonlocal call_count
    call_count += 1
    if call_count == 1:
      raise LogicError("First attempt fails")
    return "healed"

  agent = Agent(
    model="test",
    tools=[self_healing_tool],
    max_turns=5,
  )

  session = Session(session_id="heal-test", agent=agent, deps=None)
  session.memory = WorkingMemory(session_id="heal-test", memory_backend=None)

  responses = [
    LLMResponse(
      content=None,
      tool_calls=[_tool_call("self_healing_tool", {"mode": "aggressive"}, "tc1")],
    ),
    LLMResponse(
      content=None,
      tool_calls=[_tool_call("self_healing_tool", {"mode": "gentle"}, "tc2")],
    ),
    LLMResponse(content="Healed successfully."),
  ]

  mock_runner.llm.chat = AsyncMock(side_effect=responses)

  paradigm = ReActAgent(max_repeated_tool_calls=2)
  result = await paradigm.run(session, mock_runner, prompt="Heal the system")

  assert result.success is True
  loop_warnings = [
    e for e in session.memory.entries
    if e.role == MemoryRole.SYSTEM and ("loop" in e.content.lower() or "repeatedly" in e.content.lower())
  ]
  assert len(loop_warnings) == 0, "Self-healing pattern (1 fail + 1 success with different args) should not trigger loop"
