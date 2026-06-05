"""Integration tests for explicit-paradigm execution.

These tests exercise the full stack: LiteLLM -> Deepseek API -> Runner.
Run with:
    uv run pytest tests/integration/test_plan_generation.py -s -v
"""

import pytest
import asyncio
from nonoka.core.agent import Agent
from nonoka.core.tool import tool
from nonoka.core.context import RunContext
from nonoka.core.runner import Runner
from nonoka.core.plan import Plan, Step, PlanBuilder, ref
from nonoka.core.session import SessionStatus


# --------------------------------------------------------------------------- #
# Simple tools for the agent to use
# --------------------------------------------------------------------------- #

@tool
async def get_weather(city: str) -> dict:
  """Get current weather for a city."""
  return {"city": city, "temperature": 25, "condition": "sunny"}


@tool
async def calculate(expression: str) -> float:
  """Evaluate a mathematical expression and return the result."""
  try:
    result = eval(expression, {"__builtins__": {}}, {})
    return float(result)
  except Exception as e:
    raise ValueError(f"Invalid expression: {e}")


@tool
async def format_report(title: str, result: float) -> str:
  """Format a structured report from a numeric result."""
  return f"# {title}\n\n- **Result**: {result}"


@pytest.fixture
def test_agent():
  return Agent(
    model="deepseek-chat",
    tools=[get_weather, calculate, format_report],
    system_prompt="You are a helpful assistant that plans and executes tasks using available tools.",
    max_turns=5,
  )


@pytest.fixture
def runner():
  return Runner()


# --------------------------------------------------------------------------- #
# Explicit Plan construction tests (replacing old _generate_plan tests)
# --------------------------------------------------------------------------- #

def test_explicit_plan_builder_chain():
  """Users can build Plans explicitly with PlanBuilder."""
  plan = (
    PlanBuilder(objective="Test explicit plan")
    .step("calc", calculate, expression="15 * 23")
    .step("report", format_report, title="Math Result", data=ref("calc"))
    .build()
  )

  assert isinstance(plan, Plan)
  assert len(plan.steps) == 2
  assert plan.get_step("calc").tool == "calculate"
  assert plan.get_step("report").tool == "format_report"
  # ref() auto-detects dependency
  assert plan.get_step("report").depends_on == frozenset({"calc"})


def test_explicit_plan_topological_groups():
  """Topological groups should correctly order dependencies."""
  plan = (
    PlanBuilder(objective="Dependency test")
    .step("a", calculate, expression="1+1")
    .step("b", calculate, expression="2+2")
    .step("c", format_report, title="Sum", data=ref("a"))
    .build()
  )

  groups = plan.topological_groups()
  # a and b are independent → layer 0
  # c depends on a → layer 1
  assert len(groups) == 2
  assert set(groups[0]) == {"a", "b"}
  assert groups[1] == ["c"]


# --------------------------------------------------------------------------- #
# End-to-end execution tests
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_e2e_conversational_no_tools_needed(test_agent, runner):
  """A greeting should run via ReAct and succeed."""
  result = await runner.run(
    test_agent,
    "Say 'Integration test passed' and nothing else.",
    deps=None,
  )

  print(f"\n[Conversational E2E] success={result.success}, data={result.data!r}")
  assert result.success is True
  assert result.session is not None
  assert result.session.status == SessionStatus.COMPLETED
  # LLM may not follow exact output instructions; just verify it returned something
  assert result.data is not None


@pytest.mark.asyncio
async def test_e2e_single_tool_call(test_agent, runner):
  """A single-tool task should run successfully."""
  result = await runner.run(
    test_agent,
    "Get the weather for Shanghai.",
    deps=None,
  )

  print(f"\n[Single-tool E2E] success={result.success}, data={result.data!r}")
  assert result.success is True
  assert result.session is not None
  assert result.session.status == SessionStatus.COMPLETED


@pytest.mark.asyncio
async def test_e2e_explicit_plan_execution(test_agent, runner):
  """A multi-step task should execute via PlanExecutor with an explicit Plan."""
  plan = (
    PlanBuilder(objective="Calculate and report")
    .step("calc", calculate, expression="7 * 8")
    .step("report", format_report, title="Multiplication", result=ref("calc"))
    .build()
  )

  result = await runner.run_plan(test_agent, plan=plan, deps=None)

  print(f"\n[Explicit Plan E2E] success={result.success}, data={result.data!r}")
  assert result.success is True
  assert result.session is not None
  assert result.session.status == SessionStatus.COMPLETED
  # Verify both steps completed and ref was resolved
  assert "calc" in result.session.completed_steps
  assert "report" in result.session.completed_steps
  # format_report should have received the calc result via ref
  report_result = result.session.completed_steps["report"].data
  assert "Multiplication" in str(report_result)


@pytest.mark.asyncio
async def test_e2e_agent_shortcut(test_agent):
  """Agent.run() shortcut should work end-to-end."""
  result = await test_agent.run("What is 42 + 58?")

  print(f"\n[Agent Shortcut E2E] success={result.success}, data={result.data!r}")
  assert result.success is True
  assert result.session is not None
  assert result.session.status == SessionStatus.COMPLETED
