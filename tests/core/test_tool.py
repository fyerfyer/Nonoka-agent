import pytest
from nonoka.core.tool import tool
from nonoka.core.context import RunContext
from nonoka.core.session import Session
from nonoka.core.agent import Agent


def test_tool_schema_generation():
  @tool(description="Get weather")
  async def get_weather(ctx: RunContext[str], city: str, days: int = 3) -> dict:
    return {"city": city, "days": days}

  assert get_weather.name == "get_weather"
  assert get_weather.description == "Get weather"

  schema = get_weather.parameters
  assert "city" in schema["properties"]
  assert "days" in schema["properties"]

  # RunContext parameter must be excluded from the schema exposed to LLM
  assert "ctx" not in schema["properties"]

  # city has no default, so it must be required; days has a default, so it shouldn't be
  assert "city" in schema.get("required", [])
  assert "days" not in schema.get("required", [])


@pytest.mark.asyncio
async def test_tool_invoke_validation():
  @tool
  async def add(a: int, b: int) -> int:
    return a + b

  agent = Agent(model="test")
  session = Session(session_id="test", agent=agent, deps=None)
  ctx = RunContext(session)

  # Normal invocation — returns are now normalised to standard shape
  result = await add.invoke(ctx, {"a": 1, "b": 2})
  assert result == {"result": 3, "has_more": False}

  # Validation interception:
  # When LLM hallucinates and passes an uncastable parameter,
  # the framework should raise an explicit ValueError
  with pytest.raises(ValueError, match="validation failed"):
    # Pass a dict to an int parameter
    await add.invoke(ctx, {"a": 1, "b": {"wrong": "type"}})


# --------------------------------------------------------------------------- #
# Dead code removal
# --------------------------------------------------------------------------- #

def test_to_openai_schema_removed():
  """to_openai_schema should no longer be exported from nonoka.core.types."""
  import nonoka.core.types as types_module
  assert not hasattr(types_module, "to_openai_schema")


def test_tool_returns_removed():
  """Tool should not expose a ``returns`` property anymore."""
  @tool
  async def sample() -> dict:
    return {}

  assert not hasattr(sample, "returns")
