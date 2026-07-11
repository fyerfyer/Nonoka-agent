"""Built-in tools for skill management."""

from __future__ import annotations

from nonoka.core.context import RunContext
from nonoka.core.memory import MemoryRole
from nonoka.core.tool import tool
from nonoka.skills.registry import SkillRegistry


@tool
async def load_skill(ctx: RunContext, name: str) -> str:
  """Load the full guidance of a skill by name into the current context.

  Use this tool when the user asks you to apply a skill, or when you need the
  detailed guidance from a skill to complete the current task. The skill name
  must match one of the available skills listed in the system prompt.

  Args:
    name: The exact name of the skill to load.

  Returns:
    A confirmation message. The skill's system prompt and activation guidance
    are injected into the conversation context as a system message.
  """
  registry = ctx.session.agent.metadata.get("_skill_manager")
  if registry is None:
    return "Skill manager not available."

  if not isinstance(registry, SkillRegistry):
    return "Skill manager is not a SkillRegistry."

  skill = registry.get_skill(name)
  if skill is None:
    return f"Skill '{name}' not found."

  parts: list[str] = []
  if skill.system_prompt:
    parts.append(skill.system_prompt)
  if skill.activation_prompt:
    parts.append(skill.activation_prompt)

  if parts:
    content = f"[Skill '{skill.name}' loaded]\n\n" + "\n\n".join(parts)
    await ctx.session.memory.add(content, MemoryRole.SYSTEM)

  return f"Skill '{skill.name}' loaded. Use its guidance for the current task."
