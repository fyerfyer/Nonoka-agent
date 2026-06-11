from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nonoka.core.types import Capability
from nonoka.core.tool import tool as make_tool
from nonoka.config.resolver import resolve_tool_import

if TYPE_CHECKING:
  from nonoka.core.agent import Agent

_logger = logging.getLogger("nonoka.skills")


@dataclass(frozen=True)
class Skill:
  """A pre-configured capability package compatible with Claude Code skill format.

  A Skill is parsed from a Markdown file with YAML frontmatter and provides
  a declarative way to bundle tools, system prompts, and activation context.

  Usage::

    skill = Skill.from_file("skills/code-review.md")
    agent = skill.apply_to(agent)
  """
  name: str
  description: str
  tools: list[Capability] = field(default_factory=list)
  system_prompt: str = ""
  activation_prompt: str = ""
  metadata: dict[str, Any] = field(default_factory=dict)

  # ------------------------------------------------------------------ #
  # Parsing
  # ------------------------------------------------------------------ #

  @classmethod
  def from_file(cls, path: str | Path) -> Skill:
    """Parse a Skill from a Markdown file with YAML frontmatter.

    Args:
      path: Path to the ``.md`` skill file.

    Raises:
      FileNotFoundError: If the file does not exist.
      ValueError: If the file format is invalid or missing required fields.
    """
    path = Path(path)
    if not path.exists():
      raise FileNotFoundError(f"Skill file not found: {path}")
    content = path.read_text(encoding="utf-8")
    return cls.from_string(content, source=str(path))

  @classmethod
  def from_string(cls, content: str, source: str = "<string>") -> Skill:
    """Parse Skill content from a raw string.

    Args:
      content: The full Markdown content including YAML frontmatter.
      source: A human-readable source identifier for error messages.

    Raises:
      ValueError: If the frontmatter is malformed or required fields are missing.
    """
    if not content.startswith("---"):
      raise ValueError(
        f"Skill file must start with YAML frontmatter '---': {source}"
      )

    parts = content.split("---", 2)
    if len(parts) < 3:
      raise ValueError(
        f"Invalid skill file format (missing frontmatter close '---'): {source}"
      )

    import yaml
    try:
      frontmatter = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError as exc:
      raise ValueError(
        f"Invalid YAML frontmatter in {source}: {exc}"
      ) from exc

    if not isinstance(frontmatter, dict):
      raise ValueError(
        f"Skill frontmatter must be a mapping, got {type(frontmatter).__name__}: {source}"
      )

    body = parts[2].strip()

    name = frontmatter.get("name")
    if not name:
      raise ValueError(f"Skill 'name' is required in frontmatter: {source}")

    description = frontmatter.get("description", "")
    tools = cls._parse_tools(frontmatter.get("tools", []), source)

    return cls(
      name=name,
      description=description,
      tools=tools,
      system_prompt=frontmatter.get("system_prompt", ""),
      activation_prompt=body,
      metadata=frontmatter.get("metadata", {}),
    )

  @classmethod
  def _parse_tools(
    cls,
    tools_raw: list[Any],
    source: str,
  ) -> list[Capability]:
    """Resolve a list of tool declarations into Capability objects."""
    tools: list[Capability] = []
    for entry in tools_raw:
      try:
        tool = cls._resolve_tool_entry(entry, source)
        if tool is not None:
          tools.append(tool)
      except Exception as exc:
        _logger.error(f"Failed to load tool entry '{entry}' in {source}: {exc}")
    return tools

  @classmethod
  def _resolve_tool_entry(
    cls,
    entry: Any,
    source: str,
  ) -> Capability | None:
    """Resolve a single tool entry to a Capability.

    Supports:
      - ``dict`` with ``"import"`` key: ``{"import": "module:function"}``
      - ``dict`` with ``"file"`` key: ``{"file": "./path.py:function"}``
      - ``str``: ``"module:function"`` shorthand
    """
    if isinstance(entry, dict):
      if "import" in entry:
        return cls._resolve_import(entry["import"], source)
      if "file" in entry:
        return cls._resolve_file(entry["file"], source)
      _logger.warning(f"Unknown tool entry format {entry!r} in {source}")
      return None

    if isinstance(entry, str):
      return cls._resolve_import(entry, source)

    _logger.warning(f"Unsupported tool entry type {type(entry).__name__} in {source}")
    return None

  @classmethod
  def _resolve_import(cls, import_path: str, source: str) -> Capability | None:
    """Resolve ``module:function`` import path to a Capability."""
    try:
      obj = resolve_tool_import(import_path)
    except Exception as exc:
      _logger.error(f"Cannot import tool '{import_path}' from {source}: {exc}")
      return None

    if isinstance(obj, Capability):
      return obj
    if callable(obj):
      return make_tool(obj)
    _logger.warning(
      f"Tool import '{import_path}' resolved to {type(obj).__name__}, "
      f"expected a callable or Capability in {source}"
    )
    return None

  @classmethod
  def _resolve_file(cls, file_entry: str, source: str) -> Capability | None:
    """Resolve ``file: ./path.py:function`` entry to a Capability."""
    file_path, sep, func_name = file_entry.rpartition(":")
    if not sep:
      # No colon — treat whole file as auto-discover
      func_name = None

    from nonoka.core.hot_reload import PluginManager
    pm = PluginManager()
    try:
      result = pm.load_tool_from_file(file_path, func_name)
    except Exception as exc:
      _logger.error(f"Cannot load tool from file '{file_entry}' in {source}: {exc}")
      return None

    if isinstance(result, list):
      if len(result) == 1:
        return result[0]
      _logger.warning(
        f"File entry '{file_entry}' auto-discovered {len(result)} tools; "
        f"using the first one in {source}"
      )
      return result[0] if result else None
    return result

  # ------------------------------------------------------------------ #
  # Application
  # ------------------------------------------------------------------ #

  def apply_to(self, agent: Agent) -> Agent:
    """Apply this skill to *agent*, returning a new merged Agent.

    Merge rules:
      - **Tools**: Agent explicit tools have highest priority. Skill tools
        fill gaps (no overwrites).
      - **System prompt**: ``agent.system_prompt + skill.system_prompt +
        skill.activation_prompt``, joined by ``\\n\\n``.
      - **Metadata**: Skill metadata takes precedence over Agent metadata.

    The returned Agent has ``skills=[]`` because the skill has already been
    expanded into concrete fields.
    """
    from nonoka.core.agent import Agent
    from nonoka.core.hot_reload import ToolListProxy

    # Resolve current tools (may be ToolListProxy)
    current_tools: list[Capability]
    if isinstance(agent.tools, ToolListProxy):
      current_tools = list(agent.tools)
    else:
      current_tools = list(agent.tools)

    # Build merged tool map: Agent tools override skill tools
    tool_map: dict[str, Capability] = {}
    for tool in self.tools:
      tool_map[tool.name] = tool
    for tool in current_tools:
      tool_map[tool.name] = tool

    merged_tools = list(tool_map.values())

    # Merge system prompts
    parts: list[str] = []
    if agent.system_prompt:
      parts.append(agent.system_prompt)
    if self.system_prompt:
      parts.append(self.system_prompt)
    if self.activation_prompt:
      parts.append(self.activation_prompt)
    merged_system_prompt = "\n\n".join(parts)

    # Merge metadata: skill keys take precedence
    merged_metadata = dict(agent.metadata)
    merged_metadata.update(self.metadata)

    return Agent(
      model=agent.model,
      tools=merged_tools,
      system_prompt=merged_system_prompt,
      skills=[],
      deps_type=agent.deps_type,
      result_type=agent.result_type,
      max_turns=agent.max_turns,
      max_steps=agent.max_steps,
      max_concurrency=agent.max_concurrency,
      default_retry=agent.default_retry,
      default_timeout=agent.default_timeout,
      metadata=merged_metadata,
      tags=list(agent.tags),
    )
