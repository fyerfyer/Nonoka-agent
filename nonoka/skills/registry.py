"""Lazy skill registry for nonoka-agent.

A :class:`SkillRegistry` discovers available skills and exposes them to the
model as a lightweight registry block (name + description). The full content
of a skill is only loaded when the model calls the ``load_skill`` tool. This
keeps the system prompt compact while still making skill tools available for
calls that do not require the full guidance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from nonoka.core.types import Capability
from nonoka.skills.loader import SkillLoader
from nonoka.skills.skill import Skill

_logger = structlog.get_logger("nonoka.skills.registry")


@dataclass(frozen=True)
class SkillInfo:
  """Lightweight metadata for a discovered skill."""

  name: str
  description: str
  source: Path


class SkillRegistry:
  """Discover, register, and lazy-load skills.

  Search order (later overrides earlier for duplicate names):
    1. ``./skills/`` — project-level skills
    2. ``./.nonoka/skills/`` — project hidden directory
    3. ``~/.config/nonoka/skills/`` — user-level global skills
    4. Any additional ``search_paths`` provided

  Args:
    enabled: Skill names to enable. If ``None``, all discovered skills are
      considered enabled.
    search_paths: Additional directories to scan.
  """

  _STANDARD_PATHS: tuple[Path, ...] = (
    Path("skills"),
    Path(".nonoka/skills"),
    Path.home() / ".config" / "nonoka" / "skills",
  )

  def __init__(
    self,
    enabled: list[str] | None = None,
    search_paths: list[Path | str] | None = None,
  ):
    self._enabled_names: list[str] | None = enabled
    self._search_paths: list[Path] = [
      Path(p).expanduser() for p in (search_paths or [])
    ]
    self._all_paths: list[Path] = list(self._STANDARD_PATHS) + self._search_paths
    self._available: dict[str, SkillInfo] = {}
    self._loaded: dict[str, Skill] = {}

  @property
  def standard_paths(self) -> list[Path]:
    """Return the standard skill search paths."""
    return [p.expanduser() for p in self._STANDARD_PATHS]

  def discover(self) -> dict[str, SkillInfo]:
    """Scan all search paths and populate available skills."""
    if self._available:
      return dict(self._available)

    skills_by_name: dict[str, SkillInfo] = {}
    for path in self._all_paths:
      path = path.expanduser()
      if not path.exists() or not path.is_dir():
        continue
      for skill in SkillLoader(path).load_all():
        info = SkillInfo(
          name=skill.name,
          description=skill.description or "No description provided.",
          source=Path(skill.source) if skill.source else path / f"{skill.name}.md",
        )
        if skill.name in skills_by_name:
          _logger.warning(
            "skill_duplicate",
            name=skill.name,
            path=str(path),
          )
        skills_by_name[skill.name] = info

    self._available = skills_by_name
    return dict(skills_by_name)

  @property
  def available(self) -> list[SkillInfo]:
    """Return all discovered skills."""
    return list(self.discover().values())

  @property
  def enabled(self) -> list[SkillInfo]:
    """Return enabled skills.

    If no ``enabled`` list was provided, all discovered skills are returned.
    """
    all_available = self.discover()
    if self._enabled_names is None:
      return list(all_available.values())
    result: list[SkillInfo] = []
    for name in self._enabled_names:
      if name in all_available:
        result.append(all_available[name])
      else:
        _logger.warning("skill_not_found", name=name)
    return result

  def get_skill(self, name: str) -> Skill | None:
    """Load and return a skill by name.

    Loaded skills are cached so subsequent calls are cheap.
    """
    if name in self._loaded:
      return self._loaded[name]

    all_available = self.discover()
    info = all_available.get(name)
    if info is None:
      return None

    try:
      skill = SkillLoader.load_file(info.source)
    except Exception as exc:  # noqa: BLE001
      _logger.error("skill_load_failed", name=name, source=str(info.source), error=str(exc))
      return None

    self._loaded[name] = skill
    return skill

  def get_tools(self) -> list[Capability]:
    """Return tools from all enabled skills.

    Tools are registered eagerly so the model can call them without first
    loading the skill body. The skill guidance itself remains lazy.
    """
    tools: list[Capability] = []
    seen: set[str] = set()
    for info in self.enabled:
      skill = self.get_skill(info.name)
      if skill is None:
        continue
      for tool in skill.tools:
        if tool.name in seen:
          continue
        seen.add(tool.name)
        tools.append(tool)
    return tools

  def build_registry_block(self) -> str:
    """Build a system-prompt block listing available skills."""
    skills = self.enabled
    if not skills:
      return ""
    lines = [
      "## Available Skills",
      "The following skills can be referenced by name. "
      "Call the `load_skill` tool with the skill name to load its full guidance.",
    ]
    for info in skills:
      lines.append(f"- `{info.name}`: {info.description}")
    return "\n".join(lines)
