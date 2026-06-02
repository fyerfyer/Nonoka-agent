import pytest
from nonoka.core.agent import Agent
from nonoka.core.tool import tool


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
