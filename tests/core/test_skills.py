from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from nonoka import Agent, Skill, SkillLoader, tool
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

def test_skill_load_bad_file_logs_error(caplog):
  with tempfile.TemporaryDirectory() as tmpdir:
    tmp = Path(tmpdir)
    (tmp / "bad.md").write_text("not valid frontmatter")

    loader = SkillLoader(tmp)
    with caplog.at_level("ERROR", logger="nonoka.skills"):
      skills = loader.load_all()

    assert skills == []
    assert any("bad.md" in r.message for r in caplog.records)


def test_skill_parse_tool_import_failure_logs_warning(caplog):
  content = """\
---
name: bad-tools
description: Has bad tool imports
tools:
  - import: definitely.not.a.module:func
---

Body.
"""
  with caplog.at_level("ERROR", logger="nonoka.skills"):
    skill = Skill.from_string(content)

  assert skill.name == "bad-tools"
  assert skill.tools == []
  assert any("definitely.not.a.module" in r.message for r in caplog.records)
