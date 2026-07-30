"""Built-in tools for skill management."""

from __future__ import annotations

from nonoka.core.context import RunContext
from nonoka.core.tool import tool
from nonoka.core.tool_response import ToolResponse
from nonoka.skills.registry import SkillRegistry


@tool
async def load_skill(ctx: RunContext, name: str) -> ToolResponse:
  """Load the full guidance of a skill by name into the current context.

  Use this tool when the user asks you to apply a skill, or when you need the
  detailed guidance from a skill to complete the current task. The skill name
  must match one of the available skills listed in the system prompt.

  Args:
    name: The exact name of the skill to load.

  Returns:
    Structured skill content marked as durable context. Relative references
    are resolved against the returned skill directory.
  """
  registry = ctx.session.agent.metadata.get("_skill_manager")
  if registry is None:
    return ToolResponse(data="Skill manager not available.")

  if not isinstance(registry, SkillRegistry):
    return ToolResponse(data="Skill manager is not a SkillRegistry.")

  skill = registry.get_skill(name)
  if skill is None:
    return ToolResponse(data=f"Skill '{name}' is unavailable or disabled.")

  parts: list[str] = []
  if skill.system_prompt:
    parts.append(skill.system_prompt)
  if skill.activation_prompt:
    parts.append(skill.activation_prompt)

  skill_dir = skill.directory
  resources = skill.resources()
  directory_line = (
    f"Skill directory: {skill_dir}\n"
    "Resolve every relative path in this skill against that directory."
    if skill_dir is not None
    else "Skill directory: managed by the external host."
  )
  resource_block = ""
  if resources:
    resource_block = (
      "\n\n<skill_resources>\n"
      + "\n".join(f"<file>{resource}</file>" for resource in resources)
      + "\n</skill_resources>"
    )
  guidance = "\n\n".join(parts) if parts else "No additional instructions."
  content = (
    f'<skill_content name="{skill.name}">\n'
    f"{guidance}\n\n{directory_line}{resource_block}\n"
    "</skill_content>\n\n"
    f"Skill '{skill.name}' loaded. Use its guidance for the current task."
  )
  return ToolResponse(
    data=content,
    metadata={
      "context_protected": True,
      "skill_name": skill.name,
      "skill_directory": str(skill_dir) if skill_dir is not None else None,
    },
  )
