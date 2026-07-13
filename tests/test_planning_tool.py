from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from nonoka.core.agent import Agent
from nonoka.core.context import RunContext
from nonoka.core.runner import Runner
from nonoka.tools.planning import plan_task


@pytest.fixture
async def ctx():
  """A minimal RunContext bound to a test session."""
  runner = Runner(checkpoint="memory", memory="disabled")
  agent = Agent(model="test-model", tools=[])
  session = await runner._create_session(agent, deps=None)
  return RunContext(session)


@pytest.mark.asyncio
async def test_plan_task_returns_structured_plan(ctx):
  """plan_task should return a formatted string when the planner succeeds."""
  fake_plan = json.dumps({
    "goal": "Add a greeting feature",
    "steps": [
      {"id": 1, "description": "Read main.py", "target_files": ["main.py"], "tool_hint": "read_file"},
      {"id": 2, "description": "Add greeting function", "target_files": ["main.py"], "tool_hint": "write_file"},
    ],
  })

  with patch("nonoka.core.planner.Runner.run_react", new_callable=AsyncMock) as mock_run:
    mock_run.return_value = MagicMock(
      success=True,
      data=fake_plan,
      error=None,
      error_type=None,
    )
    result = await plan_task(ctx, "Add a greeting feature", max_steps=5)

  assert isinstance(result, str)
  assert "Goal: Add a greeting feature" in result
  assert "1. Read main.py" in result
  assert "[files: main.py; tool: read_file]" in result
  assert "2. Add greeting function" in result
  assert "[files: main.py; tool: write_file]" in result


@pytest.mark.asyncio
async def test_plan_task_returns_error_on_invalid_json(ctx):
  """plan_task should return an error string when the planner emits invalid JSON."""
  with patch("nonoka.core.planner.Runner.run_react", new_callable=AsyncMock) as mock_run:
    mock_run.return_value = MagicMock(
      success=True,
      data="not-json",
      error=None,
      error_type=None,
    )
    result = await plan_task(ctx, "Some task")

  assert isinstance(result, str)
  assert "Error generating plan" in result


@pytest.mark.asyncio
async def test_plan_task_falls_back_to_session_agent_model(ctx):
  """When deps has no config.model, the session agent's model is used."""
  fake_plan = json.dumps({
    "goal": "Refactor helpers",
    "steps": [
      {"id": 1, "description": "Find helper usages", "target_files": [], "tool_hint": "search"},
    ],
  })

  with patch("nonoka.core.planner.Runner.run_react", new_callable=AsyncMock) as mock_run:
    mock_run.return_value = MagicMock(
      success=True,
      data=fake_plan,
      error=None,
      error_type=None,
    )
    result = await plan_task(ctx, "Refactor helpers")

  assert "Goal: Refactor helpers" in result
  assert "1. Find helper usages" in result
