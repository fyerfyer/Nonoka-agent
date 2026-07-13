from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from nonoka.core.agent import Agent
from nonoka.core.runner import Runner


class _PlannerStep(BaseModel):
  """Schema for a single plan step."""

  id: int
  description: str
  target_files: list[str] = Field(default_factory=list)
  tool_hint: str = ""


class _PlannerOutput(BaseModel):
  """Schema for the planner's JSON output."""

  goal: str
  steps: list[_PlannerStep]


_PLANNER_SYSTEM_PROMPT = """You are a planning assistant.

Your job is to break the user's task into a small, ordered list of concrete steps.
Return ONLY a JSON object matching this exact schema (no markdown, no extra text):

{
  "goal": "<the original user task>",
  "steps": [
    {
      "id": 1,
      "description": "<clear, actionable description>",
      "target_files": ["<optional file path>"],
      "tool_hint": "<read_file|write_file|search|run_command|none>"
    }
  ]
}

Rules:
- Output valid JSON only.
- Each step must have a unique numeric id starting at 1.
- target_files is a list of file paths this step touches; use an empty list if none.
- tool_hint is a free-form hint about which tool the executor should prefer.
"""


class PlannerAgent:
  """Simple LLM-based planner that returns a validated JSON plan."""

  def __init__(self, model: str, max_steps: int = 10, max_turns: int = 3):
    self.model = model
    self.max_steps = max_steps
    self.max_turns = max_turns

  async def plan(self, task: str) -> dict[str, Any]:
    """Generate and validate a JSON plan for *task*.

    Args:
      task: The user task to plan.

    Returns:
      A validated dict with ``goal`` and ``steps``.

    Raises:
      ValueError: If the model output cannot be parsed or validated.
      RuntimeError: If the planner run itself fails (e.g. LLM error).
    """
    agent = Agent(
      model=self.model,
      system_prompt=_PLANNER_SYSTEM_PROMPT,
      max_turns=self.max_turns,
      max_steps=self.max_steps,
    )
    runner = Runner(checkpoint="memory", memory="disabled")
    result = await runner.run_react(agent, prompt=task, deps=None)

    if not result.success:
      raise RuntimeError(
        f"Planner run failed: {result.error or 'unknown error'}"
      )

    content = result.data or ""
    return _parse_and_validate_plan(content, task)


def _strip_markdown_fences(text: str) -> str:
  """Remove optional ```json ... ``` fences from model output."""
  text = text.strip()
  if text.startswith("```"):
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
  return text.strip()


def _parse_and_validate_plan(content: str, task: str) -> dict[str, Any]:
  """Parse planner output and validate it against the expected schema."""
  content = _strip_markdown_fences(content)

  try:
    data = json.loads(content)
  except json.JSONDecodeError as exc:
    raise ValueError(f"Planner output is not valid JSON: {exc}") from exc

  if not isinstance(data, dict):
    raise ValueError(f"Planner output must be a JSON object, got {type(data).__name__}")

  # Fill in the goal from the original task if the model omitted it.
  if not data.get("goal"):
    data["goal"] = task

  try:
    validated = _PlannerOutput.model_validate(data)
  except ValidationError as exc:
    raise ValueError(f"Planner output does not match expected schema: {exc}") from exc

  return validated.model_dump()
