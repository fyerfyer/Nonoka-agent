import pytest
from nonoka.core.agent import Agent
from nonoka.core.registry import ToolRegistry


def test_agent_tool_decorator():
  agent = Agent[str, str](model="gpt-4")

  @agent.tool(description="A test tool")
  async def say_hello(name: str) -> str:
    return f"Hello {name}"

  assert len(agent.tools) == 1
  assert agent.tools[0].name == "say_hello"
  assert agent.tools[0].description == "A test tool"


def test_agent_add_tools_from_registry():
  agent = Agent[str, str](model="gpt-4")
  registry = ToolRegistry()

  @registry.register(description="Registry tool")
  async def helper_tool(data: str) -> str:
    return data

  agent.add_tools(registry)
  assert len(agent.tools) == 1
  assert agent.tools[0].name == "helper_tool"
