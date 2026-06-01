import pytest
from nonoka.core.tool import tool
from nonoka.core.types import RunContext


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

  # Check return schema
  ret_schema = get_weather.returns
  assert ret_schema.get("type") == "object"


@pytest.mark.asyncio
async def test_tool_invoke_validation():
  @tool
  async def add(a: int, b: int) -> int:
    return a + b

  ctx = RunContext(deps=None, session_id="test")

  # Normal invocation
  result = await add.invoke(ctx, {"a": 1, "b": 2})
  assert result == 3

  # Validation interception:
  # When LLM hallucinates and passes an uncastable parameter,
  # the framework should raise an explicit ValueError
  with pytest.raises(ValueError, match="validation failed"):
    # Pass a dict to an int parameter
    await add.invoke(ctx, {"a": 1, "b": {"wrong": "type"}})