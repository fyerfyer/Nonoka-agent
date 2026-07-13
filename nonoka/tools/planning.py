from __future__ import annotations

from nonoka.core.context import RunContext
from nonoka.core.planner import PlannerAgent
from nonoka.core.tool import tool


def _resolve_planner_model(ctx: RunContext) -> str:
  """Pick the model for the planner agent.

  Prefer ``ctx.deps.config.model`` when available, otherwise fall back to the
  model of the agent that owns the current session.
  """
  model: str | None = None
  deps = getattr(ctx, "deps", None)
  if deps is not None:
    config = getattr(deps, "config", None)
    if config is not None:
      model = getattr(config, "model", None)
  if not model:
    session_agent = ctx.session.agent
    if session_agent is not None:
      model = session_agent.model
  if not model:
    raise RuntimeError("Cannot determine planner model: no model in deps.config or session.agent")
  return model


def _format_plan(plan: dict) -> str:
  """Render a validated plan dict as a human-readable string."""
  lines: list[str] = [f"Goal: {plan.get('goal', '')}", ""]
  steps = plan.get("steps") or []
  if not steps:
    lines.append("No steps generated.")
    return "\n".join(lines)

  lines.append("Steps:")
  for step in steps:
    step_id = step.get("id", "?")
    description = step.get("description", "")
    files = ", ".join(step.get("target_files", [])) or "none"
    hint = step.get("tool_hint") or "none"
    lines.append(f"{step_id}. {description} [files: {files}; tool: {hint}]")

  return "\n".join(lines)


@tool
async def plan_task(
  ctx: RunContext,
  task: str,
  max_steps: int = 10,
  max_turns: int = 3,
) -> str:
  """Generate a structured plan for *task* using a small planner agent.

  The planner breaks the task into numbered steps with optional target files
  and tool hints. The result is returned as a human-readable string.

  Args:
    ctx: The current tool execution context.
    task: The task to plan.
    max_steps: Maximum number of steps the planner may generate (default 10).
    max_turns: Maximum number of ReAct turns the planner may take (default 3).

  Returns:
    A formatted plan string, or an error message if planning fails.
  """
  try:
    model = _resolve_planner_model(ctx)
    planner = PlannerAgent(model=model, max_steps=max_steps, max_turns=max_turns)
    plan = await planner.plan(task)
    return _format_plan(plan)
  except Exception as exc:
    return f"Error generating plan: {exc}"
