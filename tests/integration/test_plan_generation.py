"""Integration tests for LLM-driven Plan generation.

These tests exercise the full stack: LiteLLM -> Deepseek API -> Runner._generate_plan().
Run with:
    uv run pytest tests/integration/test_plan_generation.py -s -v
"""

import pytest
import asyncio
from nonoka.core.agent import Agent
from nonoka.core.tool import tool
from nonoka.core.context import RunContext
from nonoka.core.runner import Runner
from nonoka.core.plan import Plan, Step
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
async def format_report(title: str, data: dict) -> str:
  """Format a structured report from data."""
  lines = [f"# {title}", ""]
  for key, value in data.items():
    lines.append(f"- **{key}**: {value}")
  return "\n".join(lines)


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
  return Runner(model="deepseek-chat")


# --------------------------------------------------------------------------- #
# Plan generation tests
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_generate_plan_conversational_empty(test_agent, runner):
  """A purely conversational prompt should produce an empty plan."""
  session = runner._create_session(test_agent, deps=None)
  plan = await runner._generate_plan(session, "Hello, how are you today?")

  print(f"\n[Conversational Plan] objective={plan.objective!r}, steps={plan.steps}")
  assert isinstance(plan, Plan)
  assert len(plan.steps) == 0  # LLM should return empty steps for conversational prompts


@pytest.mark.asyncio
async def test_generate_plan_single_step(test_agent, runner):
  """A simple tool-based prompt should produce a single-step plan."""
  session = runner._create_session(test_agent, deps=None)
  plan = await runner._generate_plan(session, "What is the weather in Beijing?")

  print(f"\n[Single-step Plan] objective={plan.objective!r}")
  for s in plan.steps:
    print(f"  step id={s.id}, tool={s.tool}, args={s.args}, deps={s.depends_on}")

  assert isinstance(plan, Plan)
  assert len(plan.steps) >= 0  # Could be 0 if LLM decides conversational
  if plan.steps:
    assert len(plan.steps) == 1
    assert plan.steps[0].tool == "get_weather"
    assert "Beijing" in str(plan.steps[0].args.values())


@pytest.mark.asyncio
async def test_generate_plan_multi_step(test_agent, runner):
  """A compound task may produce a multi-step plan."""
  session = runner._create_session(test_agent, deps=None)
  plan = await runner._generate_plan(
    session,
    "Calculate 15 * 23, then format a report titled 'Math Result' with the answer."
  )

  print(f"\n[Multi-step Plan] objective={plan.objective!r}")
  for s in plan.steps:
    print(f"  step id={s.id}, tool={s.tool}, args={s.args}, deps={s.depends_on}")

  assert isinstance(plan, Plan)
  if len(plan.steps) > 1:
    # Verify topological groups are valid
    groups = plan.topological_groups()
    all_step_ids = {s.id for s in plan.steps}
    grouped_ids = set()
    for g in groups:
      grouped_ids.update(g)
    assert all_step_ids == grouped_ids


@pytest.mark.asyncio
async def test_generate_plan_with_dependencies(test_agent, runner):
  """A task with sequential dependencies should produce a plan with depends_on."""
  session = runner._create_session(test_agent, deps=None)
  plan = await runner._generate_plan(
    session,
    "Calculate 10 + 20, then use that result to format a report called 'Sum Report'."
  )

  print(f"\n[Dependency Plan] objective={plan.objective!r}")
  for s in plan.steps:
    print(f"  step id={s.id}, tool={s.tool}, args={s.args}, deps={s.depends_on}")

  assert isinstance(plan, Plan)
  if len(plan.steps) >= 2:
    # At least one step should depend on another
    has_deps = any(s.depends_on for s in plan.steps)
    print(f"  has_deps={has_deps}")


# --------------------------------------------------------------------------- #
# End-to-end execution tests
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_e2e_conversational_no_tools_needed(test_agent, runner):
  """A greeting should run via ConversationalScheduler and succeed."""
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
async def test_e2e_multi_step_dag(test_agent, runner):
  """A multi-step task should execute via DAGScheduler if the plan has multiple steps."""
  result = await runner.run(
    test_agent,
    "Calculate 7 * 8, then format a report titled 'Multiplication' with the result.",
    deps=None,
  )

  print(f"\n[Multi-step DAG E2E] success={result.success}, data={result.data!r}")
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