"""Tests for host-managed external skill registry."""

from __future__ import annotations

import pytest

from nonoka import (
  AgentBuilder,
  ExternalSkill,
  ExternalSkillRegistry,
  ExternalSkillToolDefinition,
)
from nonoka.core.errors import ExternalToolExecutionRequiredError
from nonoka.core.memory import MemoryRole
from nonoka.skills.registry import SkillInfo
from nonoka.tools import load_skill


def test_external_skill_registry_discover():
  registry = ExternalSkillRegistry([
    ExternalSkill(
      name="code-review",
      description="Review code changes.",
      tools=[],
      system_prompt="You are a code reviewer.",
      activation_prompt="Review carefully.",
    ),
  ])

  discovered = registry.discover()
  assert "code-review" in discovered
  info = discovered["code-review"]
  assert isinstance(info, SkillInfo)
  assert info.name == "code-review"
  assert info.description == "Review code changes."
  assert str(info.source) == "external:code-review"


@pytest.mark.asyncio
async def test_external_skill_registry_get_skill():
  registry = ExternalSkillRegistry([
    ExternalSkill(
      name="code-review",
      description="Review code changes.",
      tools=[
        ExternalSkillToolDefinition(
          name="review_file",
          description="Review a file.",
          parameters={"type": "object", "properties": {}},
        ),
      ],
      system_prompt="You are a code reviewer.",
      activation_prompt="Review carefully.",
    ),
  ])

  skill = registry.get_skill("code-review")
  assert skill is not None
  assert skill.name == "code-review"
  assert skill.system_prompt == "You are a code reviewer."
  assert skill.activation_prompt == "Review carefully."
  assert len(skill.tools) == 1
  assert skill.tools[0].name == "skill__code-review__review_file"
  assert skill.tools[0].metadata == {
    "kind": "skill_tool",
    "skill": "code-review",
    "original_name": "review_file",
  }


@pytest.mark.asyncio
async def test_external_skill_registry_load_skill_injects_prompt():
  registry = ExternalSkillRegistry([
    ExternalSkill(
      name="code-review",
      description="Review code changes.",
      tools=[],
      system_prompt="You are a code reviewer.",
      activation_prompt="Review carefully.",
    ),
  ])

  agent = (
    AgentBuilder()
    .model("gpt-4o")
    .system_prompt("You are helpful.")
    .external_skill_registry(registry)
    .tool(load_skill)
    .build()
  )

  from nonoka import Runner
  from nonoka.core.context import RunContext

  runner = Runner(checkpoint="memory", memory="in_memory")
  session = await runner._create_session(agent, deps=None)
  ctx = RunContext(session=session)
  result = await load_skill(ctx, name="code-review")
  assert "loaded" in result
  assert "[Skill 'code-review' loaded]" in result
  assert "You are a code reviewer." in result
  assert "Review carefully." in result

  # Guidance is now returned in the tool result instead of being injected as a
  # system message, so no system memory entry is created for the skill.
  system_entries = [
    e for e in session.memory.entries if e.role == MemoryRole.SYSTEM
  ]
  assert len(system_entries) == 0


def test_external_skill_registry_get_tools():
  registry = ExternalSkillRegistry([
    ExternalSkill(
      name="greet",
      description="Greeting skill.",
      tools=[
        ExternalSkillToolDefinition("say_hello", "Say hello", {}),
        ExternalSkillToolDefinition("say_goodbye", "Say goodbye", {}),
      ],
    ),
  ])

  tools = registry.get_tools()
  assert len(tools) == 2
  names = {t.name for t in tools}
  assert names == {"skill__greet__say_hello", "skill__greet__say_goodbye"}


@pytest.mark.asyncio
async def test_external_skill_tool_invokes_as_external():
  registry = ExternalSkillRegistry([
    ExternalSkill(
      name="greet",
      description="Greeting skill.",
      tools=[
        ExternalSkillToolDefinition("say_hello", "Say hello", {}),
      ],
    ),
  ])

  cap = registry.get_tools()[0]
  with pytest.raises(ExternalToolExecutionRequiredError):
    await cap.invoke(None, {})


def test_external_skill_registry_build_registry_block():
  registry = ExternalSkillRegistry([
    ExternalSkill(name="code-review", description="Review code changes."),
    ExternalSkill(name="test", description="Write tests."),
  ])

  block = registry.build_registry_block()
  assert "## Available Skills (external)" in block
  assert "`code-review`: Review code changes." in block
  assert "`test`: Write tests." in block


def test_agent_builder_external_skill_registry():
  registry = ExternalSkillRegistry([
    ExternalSkill(name="code-review", description="Review.", tools=[]),
  ])
  agent = (
    AgentBuilder()
    .model("gpt-4o")
    .system_prompt("You are helpful.")
    .external_skill_registry(registry)
    .build()
  )

  assert agent.metadata.get("_skill_manager") is registry
  assert "## Available Skills (external)" in agent.system_prompt
