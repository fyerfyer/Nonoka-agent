"""External skill registry.

Host-managed skills: the host owns skill lifecycle and any skill-specific tool
execution; nonoka registers skill tool schemas, injects the available-skill
list into the system prompt, and loads full guidance on demand via the
``load_skill`` tool.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nonoka.core.external_tool import ExternalCapability
from nonoka.core.types import Capability
from nonoka.skills.registry import SkillInfo, SkillRegistry
from nonoka.skills.skill import Skill


def _sanitize_tool_name(name: str) -> str:
  """Return a provider-safe tool name.

  OpenAI function names must match ``^[a-zA-Z0-9_-]+$``. We replace namespace
  separators (``:``) with ``__`` and replace any remaining invalid characters
  with underscores. Double underscores are preserved as the namespace marker.
  """
  sanitized = name.replace(":", "__")
  sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", sanitized)
  return sanitized.strip("_")


@dataclass
class ExternalSkillToolDefinition:
  """A tool provided by an external skill."""

  name: str
  description: str
  parameters: dict[str, Any]


@dataclass
class ExternalSkill:
  """A host-managed skill definition."""

  name: str
  description: str
  tools: list[ExternalSkillToolDefinition] = field(default_factory=list)
  system_prompt: str = ""
  activation_prompt: str = ""


class ExternalSkillRegistry(SkillRegistry):
  """Skill registry backed by host-supplied skill definitions.

  Unlike :class:`SkillRegistry`, this registry does not scan the filesystem.
  It accepts skill definitions directly from the host and exposes them with
  the same lazy-loading contract: names/descriptions are injected into the
  system prompt, and ``load_skill`` loads the full guidance on demand.
  """

  def __init__(self, skills: list[ExternalSkill] | None = None):
    self._external_skills: dict[str, ExternalSkill] = {
      s.name: s for s in (skills or [])
    }
    # Initialize the base registry with empty search paths so discover()
    # returns only the external skills provided by the host.
    super().__init__(
      enabled=list(self._external_skills.keys()),
      search_paths=[],
    )

  def discover(self) -> dict[str, SkillInfo]:
    """Return metadata for all external skills."""
    return {
      name: SkillInfo(
        name=skill.name,
        description=skill.description or "No description provided.",
        source=Path(f"external:{skill.name}"),
      )
      for name, skill in self._external_skills.items()
    }

  @property
  def available(self) -> list[SkillInfo]:
    return list(self.discover().values())

  @property
  def enabled(self) -> list[SkillInfo]:
    return list(self.discover().values())

  def get_skill(self, name: str) -> Skill | None:
    """Load and return an external skill by name."""
    external = self._external_skills.get(name)
    if external is None:
      return None
    return Skill(
      name=external.name,
      description=external.description,
      tools=[
        ExternalCapability(
          name=_sanitize_tool_name(f"skill__{external.name}__{tool.name}"),
          description=tool.description,
          parameters=tool.parameters,
          metadata={
            "kind": "skill_tool",
            "skill": external.name,
            "original_name": tool.name,
          },
        )
        for tool in external.tools
      ],
      system_prompt=external.system_prompt,
      activation_prompt=external.activation_prompt,
      source=f"external:{external.name}",
    )

  def get_tools(self) -> list[Capability]:
    """Return tools from all external skills with per-skill namespace prefix."""
    tools: list[Capability] = []
    for skill in self._external_skills.values():
      prefix = f"skill__{skill.name}__"
      for tool in skill.tools:
        prefixed = _sanitize_tool_name(f"{prefix}{tool.name}")
        tools.append(
          ExternalCapability(
            name=prefixed,
            description=tool.description,
            parameters=tool.parameters,
            metadata={
              "kind": "skill_tool",
              "skill": skill.name,
              "original_name": tool.name,
            },
          )
        )
    return tools

  def build_registry_block(self) -> str:
    """Build a system-prompt block listing available external skills."""
    if not self._external_skills:
      return ""
    lines = [
      "## Available Skills (external)",
      "Call the `load_skill` tool with the skill name to load its full guidance.",
    ]
    for skill in self._external_skills.values():
      lines.append(f"- `{skill.name}`: {skill.description}")
    return "\n".join(lines)
