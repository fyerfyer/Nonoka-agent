from __future__ import annotations

import tempfile
import typing
from pathlib import Path

import pytest
import structlog

from nonoka import Agent, Skill, SkillLoader, SkillRegistry, tool
from nonoka.core.runtime import CompletionContract, RuntimeLimits
from nonoka.core.tool import Tool


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


@tool
async def dummy_tool_a(query: str) -> dict:
  """Tool A for testing."""
  return {"result": f"A:{query}"}


@tool
async def dummy_tool_b(value: int) -> dict:
  """Tool B for testing."""
  return {"result": f"B:{value}"}


def make_skill_file(name: str, content: str) -> Path:
  """Write a temporary skill file and return its path."""
  fd, path = tempfile.mkstemp(suffix=".md")
  with open(fd, "w", encoding="utf-8") as f:
    f.write(content)
  return Path(path)


# --------------------------------------------------------------------------- #
# Skill parsing
# --------------------------------------------------------------------------- #


def test_skill_from_string_basic():
  content = """\
---
name: code-review
description: A code review expert
system_prompt: |
  You are a meticulous code reviewer.
metadata:
  category: development
  version: "1.0.0"
---

When reviewing code:
1. Check security first
2. Check performance second
"""
  skill = Skill.from_string(content, source="test")

  assert skill.name == "code-review"
  assert skill.description == "A code review expert"
  assert skill.system_prompt == "You are a meticulous code reviewer.\n"
  assert "Check security first" in skill.activation_prompt
  assert skill.metadata == {"category": "development", "version": "1.0.0"}
  assert skill.tools == []


def test_skill_from_string_missing_name_raises():
  content = """\
---
description: Missing name
---

Body here.
"""
  with pytest.raises(ValueError, match="name"):
    Skill.from_string(content)


def test_skill_from_string_no_frontmatter_raises():
  content = "Just markdown without frontmatter."
  with pytest.raises(ValueError, match="frontmatter"):
    Skill.from_string(content)


def test_skill_from_string_unclosed_frontmatter_raises():
  content = "---\nname: foo\nJust markdown."
  with pytest.raises(ValueError, match="frontmatter"):
    Skill.from_string(content)


def test_skill_from_file():
  content = """\
---
name: data-analysis
description: Analyze datasets
---

Always provide summary statistics.
"""
  path = make_skill_file("data-analysis.md", content)
  try:
    skill = Skill.from_file(path)
    assert skill.name == "data-analysis"
    assert skill.activation_prompt == "Always provide summary statistics."
  finally:
    path.unlink()


# --------------------------------------------------------------------------- #
# Skill.apply_to
# --------------------------------------------------------------------------- #


def test_skill_apply_to_merge_tools():
  skill = Skill(
    name="test-skill",
    description="Test",
    tools=[dummy_tool_a],
  )
  agent = Agent(model="gpt-4o", tools=[dummy_tool_b])
  merged = skill.apply_to(agent)

  tool_names = {t.name for t in merged.tools}
  assert tool_names == {"dummy_tool_a", "dummy_tool_b"}


def test_skill_apply_to_agent_tools_override_skill_tools():
  """Agent explicit tools have priority over skill tools."""
  skill = Skill(
    name="test-skill",
    description="Test",
    tools=[dummy_tool_a],
  )

  # Create a different tool with the same name
  async def override_impl(query: str) -> dict:
    """Overridden tool A."""
    return {"result": f"override:{query}"}

  override_tool = Tool(override_impl)
  override_tool._name = "dummy_tool_a"

  agent = Agent(model="gpt-4o", tools=[override_tool])
  merged = skill.apply_to(agent)

  a_tools = [t for t in merged.tools if t.name == "dummy_tool_a"]
  assert len(a_tools) == 1
  assert a_tools[0].description == "Overridden tool A."


def test_skill_apply_to_merge_system_prompt():
  skill = Skill(
    name="test-skill",
    description="Test",
    system_prompt="Skill system prompt.",
    activation_prompt="Skill activation prompt.",
  )
  agent = Agent(model="gpt-4o", system_prompt="Agent system prompt.")
  merged = skill.apply_to(agent)

  assert "Agent system prompt." in merged.system_prompt
  assert "Skill system prompt." in merged.system_prompt
  assert "Skill activation prompt." in merged.system_prompt


def test_skill_apply_to_merge_metadata():
  skill = Skill(
    name="test-skill",
    description="Test",
    metadata={"category": "dev", "version": "2.0"},
  )
  agent = Agent(model="gpt-4o", metadata={"category": "other", "author": "me"})
  merged = skill.apply_to(agent)

  # Skill metadata takes precedence
  assert merged.metadata["category"] == "dev"
  assert merged.metadata["version"] == "2.0"
  # Agent-only keys are preserved
  assert merged.metadata["author"] == "me"


def test_skill_apply_to_returns_agent_with_empty_skills():
  """After apply_to, the resulting Agent has skills=[] to avoid re-expansion."""
  skill = Skill(name="test-skill", description="Test")
  agent = Agent(model="gpt-4o")
  merged = skill.apply_to(agent)

  assert merged.skills == []


def test_skill_apply_to_preserves_runtime_configuration():
  marker = object()
  agent = Agent(
    model="gpt-4o",
    temperature=0.2,
    max_tokens=123,
    runtime_limits=RuntimeLimits(max_model_turns=7),
    completion_contract=CompletionContract(),
    extensions=[marker],
  )

  merged = Skill(name="test-skill", description="Test").apply_to(agent)

  assert merged.temperature == 0.2
  assert merged.max_tokens == 123
  assert merged.runtime_limits == agent.runtime_limits
  assert merged.completion_contract == agent.completion_contract
  assert merged.extensions == [marker]


# --------------------------------------------------------------------------- #
# Agent __post_init__ with skills
# --------------------------------------------------------------------------- #


def test_agent_expands_skills_on_construction():
  skill = Skill(
    name="test-skill",
    description="Test",
    tools=[dummy_tool_a],
    system_prompt="Skill prompt.",
    metadata={"key": "value"},
  )
  agent = Agent(
    model="gpt-4o",
    tools=[dummy_tool_b],
    system_prompt="Agent prompt.",
    skills=[skill],
  )

  # Tools merged
  tool_names = {t.name for t in agent.tools}
  assert tool_names == {"dummy_tool_a", "dummy_tool_b"}

  # System prompt merged
  assert "Agent prompt." in agent.system_prompt
  assert "Skill prompt." in agent.system_prompt

  # Metadata merged
  assert agent.metadata["key"] == "value"

  # Skills cleared after expansion
  assert agent.skills == []


def test_agent_skills_override_order():
  """Later skills override earlier skills for tool names."""
  skill1 = Skill(
    name="s1",
    description="First",
    tools=[dummy_tool_a],
  )

  # Create a different tool with the same name
  async def alt_impl(query: str) -> dict:
    """Alternative A."""
    return {"alt": query}

  alt_tool = Tool(alt_impl)
  alt_tool._name = "dummy_tool_a"

  skill2 = Skill(
    name="s2",
    description="Second",
    tools=[alt_tool],
  )

  agent = Agent(model="gpt-4o", skills=[skill1, skill2])
  a_tools = [t for t in agent.tools if t.name == "dummy_tool_a"]
  assert len(a_tools) == 1
  assert a_tools[0].description == "Alternative A."


def test_agent_without_skills_unchanged():
  """Agent construction without skills works exactly as before."""
  agent = Agent(model="gpt-4o", tools=[dummy_tool_a])

  assert len(agent.tools) == 1
  assert agent.system_prompt == ""
  assert agent.skills == []


def test_agent_public_type_hints_resolve():
  hints = typing.get_type_hints(Agent)
  assert "tools" in hints
  assert "skills" in hints


# --------------------------------------------------------------------------- #
# SkillLoader
# --------------------------------------------------------------------------- #


def test_skill_loader_load_all():
  with tempfile.TemporaryDirectory() as tmpdir:
    tmp = Path(tmpdir)
    (tmp / "skill-a.md").write_text("""\
---
name: skill-a
description: Skill A
---

Body A.
""")
    (tmp / "skill-b.md").write_text("""\
---
name: skill-b
description: Skill B
---

Body B.
""")
    (tmp / "ignore.txt").write_text("not a skill")

    loader = SkillLoader(tmp)
    skills = loader.load_all()

    names = {s.name for s in skills}
    assert names == {"skill-a", "skill-b"}


def test_skill_loader_supports_agent_skills_directory_layout(tmp_path: Path):
  skill_dir = tmp_path / "code-review"
  skill_dir.mkdir()
  (skill_dir / "SKILL.md").write_text(
    "---\nname: code-review\ndescription: Review code.\n---\nFollow references/checklist.md.\n",
    encoding="utf-8",
  )

  skills = SkillLoader(tmp_path).load_all()

  assert [skill.name for skill in skills] == ["code-review"]
  assert skills[0].source == str((skill_dir / "SKILL.md").resolve())


def test_standard_skill_lists_bundled_resources(tmp_path: Path):
  skill_dir = tmp_path / "code-review"
  (skill_dir / "references").mkdir(parents=True)
  (skill_dir / "scripts").mkdir()
  (skill_dir / "SKILL.md").write_text(
    "---\nname: code-review\ndescription: Review code.\n---\nUse the checklist.\n",
    encoding="utf-8",
  )
  (skill_dir / "references" / "checklist.md").write_text("Check safety.")
  (skill_dir / "scripts" / "scan.py").write_text("print('ok')")

  skill = SkillLoader(tmp_path).load_all()[0]

  assert skill.directory == skill_dir.resolve()
  assert skill.resources() == ["scripts/scan.py", "references/checklist.md"]


def test_registry_discovery_does_not_import_skill_tools(tmp_path: Path):
  (tmp_path / "safe-discovery.md").write_text(
    "---\nname: safe-discovery\ndescription: Discover safely.\n"
    "tools:\n  - import: definitely.not.a.module:func\n---\nBody.\n",
    encoding="utf-8",
  )
  registry = SkillRegistry(enabled=["safe-discovery"], search_paths=[tmp_path])

  with structlog.testing.capture_logs() as logs:
    assert registry.enabled[0].name == "safe-discovery"

  assert not any("definitely.not.a.module" in str(event) for event in logs)


def test_project_agent_skills_override_user_skills(tmp_path: Path, monkeypatch):
  fake_home = tmp_path / "home"
  project = tmp_path / "project"
  user_skill = fake_home / ".agents" / "skills" / "shared"
  project_skill = project / ".agents" / "skills" / "shared"
  user_skill.mkdir(parents=True)
  project_skill.mkdir(parents=True)
  (user_skill / "SKILL.md").write_text(
    "---\nname: shared\ndescription: User version.\n---\nUser.\n",
    encoding="utf-8",
  )
  (project_skill / "SKILL.md").write_text(
    "---\nname: shared\ndescription: Project version.\n---\nProject.\n",
    encoding="utf-8",
  )
  monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
  monkeypatch.chdir(project)

  registry = SkillRegistry(enabled=["shared"])

  assert registry.enabled[0].description == "Project version."


def test_project_agent_skills_override_legacy_project_skills(tmp_path: Path, monkeypatch):
  project = tmp_path / "project"
  standard_skill = project / ".agents" / "skills" / "shared"
  legacy_skills = project / "skills"
  standard_skill.mkdir(parents=True)
  legacy_skills.mkdir(parents=True)
  (standard_skill / "SKILL.md").write_text(
    "---\nname: shared\ndescription: Standard version.\n---\nStandard.\n",
    encoding="utf-8",
  )
  (legacy_skills / "shared.md").write_text(
    "---\nname: shared\ndescription: Legacy version.\n---\nLegacy.\n",
    encoding="utf-8",
  )
  monkeypatch.chdir(project)

  registry = SkillRegistry(enabled=["shared"])

  assert registry.enabled[0].description == "Standard version."


def test_registry_cannot_load_disabled_skill(tmp_path: Path):
  (tmp_path / "disabled.md").write_text(
    "---\nname: disabled\ndescription: Disabled.\n---\nDo not load.\n",
    encoding="utf-8",
  )
  registry = SkillRegistry(enabled=[], search_paths=[tmp_path])

  assert registry.enabled == []
  assert registry.get_skill("disabled") is None


def test_skill_loader_load_file():
  content = """\
---
name: single
description: Single skill
---

Only one.
"""
  path = make_skill_file("single.md", content)
  try:
    skill = SkillLoader.load_file(path)
    assert skill.name == "single"
  finally:
    path.unlink()


def test_skill_loader_empty_directory():
  with tempfile.TemporaryDirectory() as tmpdir:
    loader = SkillLoader(tmpdir)
    skills = loader.load_all()
    assert skills == []


def test_skill_loader_missing_directory():
  loader = SkillLoader("/nonexistent/path/to/skills")
  skills = loader.load_all()
  assert skills == []


def test_skill_loader_no_path_raises():
  loader = SkillLoader()
  with pytest.raises(ValueError, match="path"):
    loader.load_all()


# --------------------------------------------------------------------------- #
# Error handling
# --------------------------------------------------------------------------- #


def test_skill_load_bad_file_logs_error():
  with tempfile.TemporaryDirectory() as tmpdir:
    tmp = Path(tmpdir)
    (tmp / "bad.md").write_text("not valid frontmatter")

    loader = SkillLoader(tmp)
    with structlog.testing.capture_logs() as cap_logs:
      skills = loader.load_all()

    assert skills == []
    messages = " ".join(str(e.get("event", "")) for e in cap_logs)
    assert "bad.md" in messages


def test_skill_parse_tool_import_failure_logs_warning():
  content = """\
---
name: bad-tools
description: Has bad tool imports
tools:
  - import: definitely.not.a.module:func
---

Body.
"""
  with structlog.testing.capture_logs() as cap_logs:
    skill = Skill.from_string(content)

  assert skill.name == "bad-tools"
  assert skill.tools == []
  messages = " ".join(str(e.get("event", "")) for e in cap_logs)
  assert "definitely.not.a.module" in messages
