from __future__ import annotations

from pathlib import Path
import structlog

from nonoka.skills.skill import Skill

_logger = structlog.get_logger("nonoka.skills")


class SkillLoader:
  """Discover Agent Skills directories and legacy flat Markdown skills.

  Usage::

    # Load from a specific directory
    loader = SkillLoader("skills/")
    skills = loader.load_all()

    # Auto-discover across standard paths
    skills = SkillLoader.auto_find()

    # Load a single file
    skill = SkillLoader.load_file("skills/code-review.md")
  """

  def __init__(self, path: str | Path | None = None):
    """Create a loader bound to a directory.

    Args:
      path: Directory to scan. If ``None``, the loader must be used with
        explicit methods (e.g. ``auto_find``).
    """
    self.path = Path(path) if path else None

  # ------------------------------------------------------------------ #
  # Instance methods
  # ------------------------------------------------------------------ #

  def load_all(self) -> list[Skill]:
    """Load every ``*.md`` skill file in the bound directory.

    Returns:
      A list of Skill objects. Files that fail to parse are logged and
      skipped (they do not stop the loading of other files).
    """
    if self.path is None:
      raise ValueError(
        "SkillLoader was created without a path. "
        "Pass a path to the constructor or use SkillLoader.auto_find()."
      )
    if not self.path.exists():
      _logger.warning(f"Skill directory not found: {self.path}")
      return []
    return self._scan_directory(self.path)

  # ------------------------------------------------------------------ #
  # Class methods
  # ------------------------------------------------------------------ #

  @classmethod
  def load_file(cls, path: str | Path) -> Skill:
    """Load a single skill file.

    Args:
      path: Path to the ``.md`` skill file.

    Returns:
      The parsed Skill.
    """
    return Skill.from_file(path)

  @classmethod
  def auto_find(cls) -> list[Skill]:
    """Auto-discover skills across standard paths.

    Search order is low-to-high precedence so project skills override user
    skills, matching the Agent Skills client convention.

    Returns:
      A deduplicated list of Skill objects. Duplicate names are resolved
      by keeping the skill from the highest-priority path.
    """
    skills_by_name: dict[str, Skill] = {}

    search_paths = cls._search_paths()
    for path in search_paths:
      if not path.exists():
        continue
      for skill in cls._scan_directory(path):
        if skill.name in skills_by_name:
          _logger.warning(
            f"Duplicate skill '{skill.name}' found in {path}; "
            f"overwriting with higher-priority version."
          )
        skills_by_name[skill.name] = skill

    return list(skills_by_name.values())

  # ------------------------------------------------------------------ #
  # Internal helpers
  # ------------------------------------------------------------------ #

  @classmethod
  def _search_paths(cls) -> list[Path]:
    """Return standard skill search paths in priority order."""
    home = Path.home()
    return [
      home / ".config" / "nonoka" / "skills",
      home / ".agents" / "skills",
      Path("skills"),
      Path(".nonoka/skills"),
      Path(".agents/skills"),
    ]

  @classmethod
  def discover_files(cls, directory: Path) -> list[Path]:
    """Return standard ``SKILL.md`` and legacy flat skill files."""
    if not directory.is_dir():
      return []
    candidates: list[Path] = []
    direct = directory / "SKILL.md"
    if direct.is_file():
      candidates.append(direct)
    candidates.extend(sorted(directory.glob("*/SKILL.md")))
    candidates.extend(path for path in sorted(directory.glob("*.md")) if path.name != "SKILL.md")
    return candidates

  @classmethod
  def _scan_directory(cls, directory: Path) -> list[Skill]:
    """Scan a single directory for ``*.md`` skill files."""
    skills: list[Skill] = []
    if not directory.is_dir():
      return skills

    for file_path in cls.discover_files(directory):
      try:
        skill = Skill.from_file(file_path)
        skills.append(skill)
      except Exception as exc:
        _logger.error(f"Failed to load skill file {file_path}: {exc}")
    return skills
