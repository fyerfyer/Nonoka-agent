import pytest
from nonoka.core.agent import Agent
from nonoka.core.tool import tool
from nonoka.core.registry import ToolRegistry


def test_agent_frozen_immutable():
  """Agent is frozen; tools must be passed at construction time."""
  @tool
  async def say_hello(name: str) -> str:
    return f"Hello {name}"

  agent = Agent[str, str](model="gpt-4", tools=[say_hello])

  assert len(agent.tools) == 1
  assert agent.tools[0].name == "say_hello"


def test_agent_tool_via_decorator_then_constructor():
  """Users create tools with @tool and pass them to Agent explicitly."""
  @tool(description="A test tool")
  async def helper(data: str) -> str:
    return data

  agent = Agent[str, str](model="gpt-4", tools=[helper])
  assert len(agent.tools) == 1
  assert agent.tools[0].name == "helper"
  assert agent.tools[0].description == "A test tool"


# --------------------------------------------------------------------------- #
# ToolRegistry -> Agent
# --------------------------------------------------------------------------- #

def test_agent_accepts_tool_registry_directly():
  """Agent(tools=registry) should expand registry tools."""
  registry = ToolRegistry()

  @registry.register
  async def tool_a(x: int) -> int:
    return x * 2

  @registry.register
  async def tool_b(y: str) -> str:
    return y.upper()

  agent = Agent(model="test", tools=registry)

  assert isinstance(agent.tools, list)
  assert len(agent.tools) == 2
  names = {t.name for t in agent.tools}
  assert names == {"tool_a", "tool_b"}


def test_agent_accepts_mixed_tools_and_registries():
  """Agent can mix raw tools and registries."""
  registry = ToolRegistry()

  @registry.register
  async def reg_tool(x: int) -> int:
    return x

  @tool
  async def direct_tool(y: str) -> str:
    return y

  agent = Agent(model="test", tools=[direct_tool, registry])
  names = {t.name for t in agent.tools}
  assert names == {"direct_tool", "reg_tool"}