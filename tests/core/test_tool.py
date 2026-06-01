from typing import Any
from nonoka.core import RetryPolicy
from pydantic import BaseModel
from nonoka.core import tool, RunContext
import pytest


class MockDatabase:
  def __init__(self):
    self.query_count = 0

  def fetch_user(self, user_id: str) -> str:
    """fetch user info"""
    self.query_count += 1
    return f"user_data_for_{user_id}"


class AppDeps:
  def __init__(self, db: MockDatabase):
    self.db = db


@tool
async def calculate_sum(a: int, b: int) -> int:
  """calculate sum"""
  return a + b


@tool(description="Get user profile", default_retry=RetryPolicy(max_retries=5))
async def get_user_profile(ctx: RunContext[AppDeps], user_id: str) -> dict[str, Any]:
  data = ctx.deps.db.fetch_user(user_id)
  return {"id": user_id, "data": data, "session": ctx.session_id}


class WeatherResult(BaseModel):
  city: str
  temperature: float


@tool
async def get_weather(city: str) -> WeatherResult:
  return WeatherResult(city=city, temperature=25.5)


@pytest.mark.asyncio
async def test_tool_invoke_without_ctx():
  dummy_ctx = RunContext(deps=None)
  result = await calculate_sum.invoke(ctx=dummy_ctx, arguments={"a": 10, "b": 20})
  assert result == 30


@pytest.mark.asyncio
async def test_tool_invoke_with_ctx():
  mock_db = MockDatabase()
  deps = AppDeps(db=mock_db)
  ctx = RunContext(deps=deps, session_id="test_sess_123")

  result = await get_user_profile.invoke(ctx=ctx, arguments={"user_id": "9527"})

  assert result["id"] == "9527"
  assert result["data"] == "user_data_for_9527"
  assert result["session"] == "test_sess_123"

  assert mock_db.query_count == 1


def test_tool_schema_ignores_ctx():
  schema = get_user_profile.to_json_schema()

  params = schema["function"]["parameters"]

  assert "user_id" in params["properties"]
  assert "user_id" in params["required"]

  assert "ctx" not in params["properties"]


def test_tool_standard_schema_generation():
  schema = calculate_sum.to_json_schema()

  assert schema["type"] == "function"
  func = schema["function"]

  assert func["name"] == "calculate_sum"
  assert func["description"] == "calculate sum"

  assert func["parameters"]["properties"]["a"]["type"] == "integer"
  assert func["parameters"]["properties"]["b"]["type"] == "integer"


def test_tool_complex_return_schema():
  returns = get_weather.returns

  assert returns["type"] == "object"
  assert returns["properties"]["city"]["type"] == "string"
  assert returns["properties"]["temperature"]["type"] == "number"


def test_tool_async_check():
  with pytest.raises(TypeError, match="must be an async function"):
    @tool
    def sync_func(a: int):
      return a


@pytest.mark.asyncio
async def test_tool_pydantic_validation_and_conversion():
  dummy_ctx = RunContext(deps=None)
  result = await calculate_sum.invoke(ctx=dummy_ctx, arguments={"a": "10", "b": "20"})
  assert result == 30
  
  # Missing required arguments should raise a ValueError
  with pytest.raises(ValueError, match="validation failed"):
    await calculate_sum.invoke(ctx=dummy_ctx, arguments={"a": 10})


@tool
async def ctx_by_type_hint(my_context: RunContext[AppDeps], data: str) -> str:
  return my_context.session_id + "_" + data


@pytest.mark.asyncio
async def test_tool_ctx_by_type_hint():
  ctx = RunContext(deps=None, session_id="test_type_hint")
  result = await ctx_by_type_hint.invoke(ctx=ctx, arguments={"data": "works"})
  assert result == "test_type_hint_works"