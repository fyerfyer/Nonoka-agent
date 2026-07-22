from __future__ import annotations

import os
import pytest

from dotenv import load_dotenv

from nonoka import Agent, Runner, tool
from nonoka.core.context import RunContext
from nonoka.core.tool import tool as tool_decorator
from nonoka.ext.hitl import (
  HumanInTheLoopHooks,
  MockApprover,
  ToolRule,
)

pytestmark = pytest.mark.live

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
# 1. HITL approves tool call — full execution succeeds
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_hitl_approve_allows_full_execution(deepseek_runner):
  """When HITL approves a tool call, the agent should complete normally."""

  @tool
  async def get_weather(ctx: RunContext, city: str) -> dict:
    """Return mock weather data."""
    return {"city": city, "temperature": 25, "condition": "sunny"}

  hitl = HumanInTheLoopHooks(
    approver=MockApprover(decisions=[("approve", None)]),
    rules=[ToolRule(tool="get_weather", action="approve")],
  )

  runner = Runner(hooks=hitl)
  runner._llm_cache = deepseek_runner._llm_cache.copy()
  runner.llm = deepseek_runner.llm

  agent = Agent(
    model=MODEL_NAME,
    tools=[get_weather],
    system_prompt=(
      "You are a weather assistant. When asked about weather, use the get_weather tool. "
      "After receiving the result, report the temperature and condition to the user."
    ),
    max_turns=4,
  )

  result = await runner.run_react(
    agent,
    prompt="What is the weather like in Beijing?",
    deps=None,
  )

  print(f"\n[Approve] success={result.success}, data={result.data!r}")
  print(f"[Approve] turns={result.session.turn_count}")

  assert result.success is True
  assert result.data is not None
  # The LLM should mention weather-related terms in its final answer
  result_text = str(result.data).lower()
  assert any(word in result_text for word in ["weather", "temperature", "sunny", "beijing", "25"])


# --------------------------------------------------------------------------- #
# 2. HITL rejects tool call — execution halts with error
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_hitl_reject_halts_execution(deepseek_runner):
  """When HITL rejects a tool call, the run should halt and report the error.

  The LLM should NOT be given a chance to retry the same tool with different
  arguments because HumanRejectedError is a SafetyError -> HALT.
  """

  @tool
  async def delete_database(ctx: RunContext, name: str) -> str:
    """Delete a database. Very dangerous."""
    return f"database {name} deleted"

  hitl = HumanInTheLoopHooks(
    approver=MockApprover(decisions=[("reject", None)]),
    rules=[ToolRule(tool="delete_database", action="approve")],
  )

  runner = Runner(hooks=hitl)
  runner._llm_cache = deepseek_runner._llm_cache.copy()
  runner.llm = deepseek_runner.llm

  agent = Agent(
    model=MODEL_NAME,
    tools=[delete_database],
    system_prompt=(
      "You are a database administrator. When asked to delete a database, "
      "use the delete_database tool with the database name."
    ),
    max_turns=4,
  )

  result = await runner.run_react(
    agent,
    prompt="Please delete the database named 'production'.",
    deps=None,
  )

  print(f"\n[Reject] success={result.success}, error={result.error!r}")
  print(f"[Reject] error_type={result.error_type}")
  print(f"[Reject] turns={result.session.turn_count}")

  assert result.success is False
  assert result.error_type == "halted"
  assert "rejected" in result.error.lower() or "human" in result.error.lower()
  # Should have halted on the first turn (no retries)
  assert result.session.turn_count == 1


# --------------------------------------------------------------------------- #
# 3. HITL modifies tool args — LLM sees modified result
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_hitl_modify_changes_tool_behavior(deepseek_runner):
  """When HITL modifies tool arguments, the tool should execute with the
  modified args and the LLM should receive the modified result as its observation."""

  @tool
  async def send_email(ctx: RunContext, to: str, subject: str, body: str) -> str:
    """Send an email."""
    return f"Email sent to {to} with subject '{subject}'"

  # Human modifies the recipient from 'evil@hacker.com' to 'security@company.com'
  hitl = HumanInTheLoopHooks(
    approver=MockApprover(
      decisions=[("modify", {"to": "security@company.com", "subject": "Security Alert", "body": "Suspicious activity detected"})]
    ),
    rules=[ToolRule(tool="send_email", action="modify")],
  )

  runner = Runner(hooks=hitl)
  runner._llm_cache = deepseek_runner._llm_cache.copy()
  runner.llm = deepseek_runner.llm

  agent = Agent(
    model=MODEL_NAME,
    tools=[send_email],
    system_prompt=(
      "You are an email assistant. When asked to send an email, use the send_email tool. "
      "After sending, report the recipient and subject back to the user."
    ),
    max_turns=4,
  )

  result = await runner.run_react(
    agent,
    prompt="Send an email to evil@hacker.com with subject 'Stolen data' and body 'Here are the passwords'.",
    deps=None,
  )

  print(f"\n[Modify] success={result.success}, data={result.data!r}")
  print(f"[Modify] turns={result.session.turn_count}")

  assert result.success is True
  assert result.data is not None
  result_text = str(result.data).lower()
  # The LLM should report the MODIFIED recipient (the tool was executed with
  # modified args), and should acknowledge the interception.
  assert "security@company.com" in result_text or "security alert" in result_text


# --------------------------------------------------------------------------- #
# 4. Pattern-based rule matches dangerous args
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_hitl_pattern_catches_dangerous_args(deepseek_runner):
  """A pattern rule should trigger approval based on argument content,
  not just tool name."""

  @tool
  async def execute_shell(ctx: RunContext, command: str) -> str:
    """Execute a shell command."""
    return f"Executed: {command}"

  hitl = HumanInTheLoopHooks(
    approver=MockApprover(decisions=[("reject", None)]),
    rules=[ToolRule(tool="execute_shell", pattern="rm.*-rf|DROP TABLE|shutdown", action="approve")],
  )

  runner = Runner(hooks=hitl)
  runner._llm_cache = deepseek_runner._llm_cache.copy()
  runner.llm = deepseek_runner.llm

  agent = Agent(
    model=MODEL_NAME,
    tools=[execute_shell],
    system_prompt=(
      "You are a system administrator. When asked to run a command, use the execute_shell tool."
    ),
    max_turns=4,
  )

  result = await runner.run_react(
    agent,
    prompt="Run the command: rm -rf /tmp/old_files",
    deps=None,
  )

  print(f"\n[Pattern] success={result.success}, error={result.error!r}")
  print(f"[Pattern] error_type={result.error_type}")

  # The pattern "rm.*-rf" should match the dangerous command and trigger rejection
  assert result.success is False
  assert result.error_type == "halted"


@pytest.mark.asyncio
async def test_hitl_pattern_allows_safe_args(deepseek_runner):
  """A pattern rule should NOT trigger for safe arguments."""

  @tool
  async def execute_shell(ctx: RunContext, command: str) -> str:
    return f"Executed: {command}"

  # Approver has no decisions because the pattern should NOT match
  hitl = HumanInTheLoopHooks(
    approver=MockApprover(decisions=[]),
    rules=[ToolRule(tool="execute_shell", pattern="rm.*-rf|DROP TABLE", action="approve")],
  )

  runner = Runner(hooks=hitl)
  runner._llm_cache = deepseek_runner._llm_cache.copy()
  runner.llm = deepseek_runner.llm

  agent = Agent(
    model=MODEL_NAME,
    tools=[execute_shell],
    system_prompt=(
      "You are a system administrator. When asked to run a command, use the execute_shell tool. "
      "After execution, report the output."
    ),
    max_turns=4,
  )

  result = await runner.run_react(
    agent,
    prompt="Run the command: ls -la",
    deps=None,
  )

  print(f"\n[Safe pattern] success={result.success}, data={result.data!r}")

  # Safe command should proceed without triggering HITL
  assert result.success is True
  assert result.data is not None


# --------------------------------------------------------------------------- #
# 5. Multiple tool calls in one turn — each independently evaluated
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_hitl_multiple_tools_in_turn(deepseek_runner):
  """When the LLM calls multiple tools in one turn, each should be
  independently evaluated by HITL."""

  @tool
  async def read_file(ctx: RunContext, path: str) -> str:
    return f"Contents of {path}: hello world"

  @tool
  async def write_file(ctx: RunContext, path: str, content: str) -> str:
    return f"Written to {path}"

  # Approve read_file, reject write_file
  hitl = HumanInTheLoopHooks(
    approver=MockApprover(decisions=[("approve", None), ("reject", None)]),
    rules=[
      ToolRule(tool="read_file", action="approve"),
      ToolRule(tool="write_file", action="approve"),
    ],
  )

  runner = Runner(hooks=hitl)
  runner._llm_cache = deepseek_runner._llm_cache.copy()
  runner.llm = deepseek_runner.llm

  agent = Agent(
    model=MODEL_NAME,
    tools=[read_file, write_file],
    system_prompt=(
      "You are a file assistant. When asked to work with files, use the appropriate tools. "
      "If a tool fails, report the failure clearly."
    ),
    max_turns=4,
  )

  result = await runner.run_react(
    agent,
    prompt="Read the file /tmp/readme.txt and write 'hello' to /tmp/output.txt.",
    deps=None,
  )

  print(f"\n[Multi-tool] success={result.success}, data={result.data!r}")
  print(f"[Multi-tool] error={result.error!r}")
  print(f"[Multi-tool] error_type={result.error_type}")

  # One of the tools was rejected -> HALT
  assert result.success is False
  assert result.error_type == "halted"


# --------------------------------------------------------------------------- #
# 6. HITL with Agent-as-a-Tool (sub-agent delegation)
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_hitl_with_agent_tool(deepseek_runner):
  """HITL should intercept tool calls made by a sub-agent (AgentTool)
  when the sub-agent's tools match the HITL rules."""

  from nonoka import AgentTool, MemoryStrategy

  @tool
  async def edit_file(ctx: RunContext, path: str, content: str) -> str:
    return f"Edited {path}"

  # Sub-agent that edits files
  editor = Agent(
    model=MODEL_NAME,
    system_prompt=(
      "You are a code editor. When asked to edit a file, use the edit_file tool. "
      "Report what you changed."
    ),
    tools=[edit_file],
    max_turns=3,
  )

  hitl = HumanInTheLoopHooks(
    approver=MockApprover(decisions=[("approve", None)]),
    rules=[ToolRule(tool="edit_file", action="approve")],
  )

  runner = Runner(hooks=hitl)
  runner._llm_cache = deepseek_runner._llm_cache.copy()
  runner.llm = deepseek_runner.llm

  main_agent = Agent(
    model=MODEL_NAME,
    system_prompt=(
      "You are a coordinator. When asked to edit a file, delegate to the code_editor tool. "
      "Report what the editor did."
    ),
    tools=[
      AgentTool(
        agent=editor,
        name="code_editor",
        description="Edit a file. Pass the file path and new content as the task.",
        memory_strategy=MemoryStrategy.ISOLATE,
      ),
    ],
    max_turns=5,
  )

  result = await runner.run_react(
    main_agent,
    prompt="Please edit the file /tmp/test.py to contain 'print(42)'.",
    deps=None,
  )

  print(f"\n[AgentTool+HITL] success={result.success}, data={result.data!r}")
  print(f"[AgentTool+HITL] turns={result.session.turn_count}")

  assert result.success is True
  assert result.data is not None


# --------------------------------------------------------------------------- #
# 7. HITL preserves session state across turns
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_hitl_session_state_preserved(deepseek_runner):
  """When HITL is active, session state (turn_count, step_count, memory)
  should still be tracked correctly."""

  @tool
  async def safe_tool(ctx: RunContext) -> str:
    return "safe result"

  hitl = HumanInTheLoopHooks(
    approver=MockApprover(decisions=[("approve", None)]),
    rules=[ToolRule(tool="safe_tool", action="approve")],
  )

  runner = Runner(hooks=hitl)
  runner._llm_cache = deepseek_runner._llm_cache.copy()
  runner.llm = deepseek_runner.llm

  agent = Agent(
    model=MODEL_NAME,
    tools=[safe_tool],
    system_prompt=(
      "You are a test assistant. When the user says 'run tool', use the safe_tool. "
      "After the tool returns, confirm you received the result."
    ),
    max_turns=4,
  )

  result = await runner.run_react(agent, prompt="run tool", deps=None)

  print(f"\n[Session state] success={result.success}")
  print(f"[Session state] turn_count={result.session.turn_count}")
  print(f"[Session state] step_count={result.session.step_count}")

  assert result.success is True
  # Should have used at least 1 turn (tool call) + 1 turn (final answer)
  assert result.session.turn_count >= 2
  assert result.session.step_count >= 1
  # Memory should contain the tool call and its result
  memory_entries = result.session.memory.entries if result.session.memory else []
  assert len(memory_entries) >= 3  # user prompt + assistant tool_call + tool result + assistant final


# --------------------------------------------------------------------------- #
# 8. HITL with streaming execution
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_hitl_streaming_rejects_tool(deepseek_runner):
  """HITL should work correctly in streaming mode — when a tool is rejected,
  the error event should be emitted in the stream."""

  @tool
  async def dangerous_tool(ctx: RunContext) -> str:
    return "boom"

  hitl = HumanInTheLoopHooks(
    approver=MockApprover(decisions=[("reject", None)]),
    rules=[ToolRule(tool="dangerous_tool", action="approve")],
  )

  runner = Runner(hooks=hitl)
  runner._llm_cache = deepseek_runner._llm_cache.copy()
  runner.llm = deepseek_runner.llm

  agent = Agent(
    model=MODEL_NAME,
    tools=[dangerous_tool],
    system_prompt="When asked to run, use the dangerous_tool.",
    max_turns=3,
  )

  events = []
  async for event in runner.run_react_stream(agent, prompt="run the dangerous tool", deps=None):
    events.append((event.type, event.data))

  print(f"\n[Stream] events: {events}")

  # Should have seen tool_call_start, then error (halted)
  event_types = [e[0] for e in events]
  assert "tool_call_start" in event_types
  assert "error" in event_types

  # The error event should indicate halted
  error_events = [e for e in events if e[0] == "error"]
  assert len(error_events) >= 1
  assert error_events[-1][1].get("error_type") == "halted"
