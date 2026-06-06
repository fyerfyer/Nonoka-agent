import pytest
from nonoka.core.agent import Agent
from nonoka.core.tool import tool


class TestAgentMetadata:
  def test_agent_default_metadata_and_tags(self):
    """Agent should have empty metadata and tags by default."""
    agent = Agent(model="gpt-4")
    assert agent.metadata == {}
    assert agent.tags == []

  def test_agent_with_metadata(self):
    """Agent should accept metadata dict."""
    agent = Agent(
      model="gpt-4",
      metadata={
        "version": "1.0",
        "team": "platform",
        "cost_center": "ai-infra",
      }
    )
    assert agent.metadata["version"] == "1.0"
    assert agent.metadata["team"] == "platform"

  def test_agent_with_tags(self):
    """Agent should accept tags list."""
    agent = Agent(
      model="gpt-4",
      tags=["production", "critical", "v2"],
    )
    assert "production" in agent.tags
    assert "critical" in agent.tags
    assert len(agent.tags) == 3

  def test_agent_frozen_with_metadata(self):
    """Agent with metadata should still be frozen (no attribute reassignment)."""
    agent = Agent(
      model="gpt-4",
      metadata={"key": "value"},
      tags=["tag1"],
    )
    # Frozen dataclass prevents attribute reassignment via normal assignment
    with pytest.raises(AttributeError):
      agent.metadata = {}
    with pytest.raises(AttributeError):
      agent.tags = []

  def test_agent_metadata_routing_use_case(self):
    """Simulate Gateway routing use case: filter by tags and metadata."""
    weather_agent = Agent(
      model="gpt-4o",
      tags=["weather", "public"],
      metadata={
        "rate_limit": 100,
        "cost_per_request": 0.02,
      },
    )

    # Simulate router logic
    def can_route(agent, required_tag):
      return required_tag in agent.tags

    def get_rate_limit(agent):
      return agent.metadata.get("rate_limit", float("inf"))

    assert can_route(weather_agent, "weather")
    assert not can_route(weather_agent, "private")
    assert get_rate_limit(weather_agent) == 100

  def test_agent_mixed_tools_registry_and_metadata(self):
    """Agent should support both ToolRegistry expansion and metadata."""
    from nonoka.core.registry import ToolRegistry

    registry = ToolRegistry()

    @registry.register
    async def get_weather(city: str) -> str:
      return f"Weather in {city}"

    agent = Agent(
      model="gpt-4o",
      tools=registry,
      metadata={"category": "weather"},
      tags=["public"],
    )

    assert len(agent.tools) == 1
    assert agent.metadata["category"] == "weather"
    assert "public" in agent.tags
