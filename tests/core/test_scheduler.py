import pytest
import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from nonoka.core.agent import Agent
from nonoka.core.tool import tool
from nonoka.core.context import RunContext
from nonoka.core.session import Session, SessionStatus, StepStatus, StepResult, StepFailure
from nonoka.core.plan import Plan, Step, PlanBuilder, ref, Ref
from nonoka.core.scheduler import (
  _resolve_refs,
  _resolve_path,
)
from nonoka.core.paradigm import ReActAgent, PlanExecutor, ToolEvaluator, EvaluationResult
from nonoka.core.errors import ErrorPolicy, TransientError, LogicError, CancelledError
from nonoka.core.types import RunResult
from nonoka.backends.checkpoint.memory import MemoryCheckpointStore


# --------------------------------------------------------------------------- #
# Ref resolution helpers
# --------------------------------------------------------------------------- #

def test_resolve_path_dict():
  data = {"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]}
  assert _resolve_path(data, "users") == data["users"]
  assert _resolve_path(data, "users[0]") == {"id": 1, "name": "Alice"}
  assert _resolve_path(data, "users[1].name") == "Bob"
  assert _resolve_path(data, "missing") is None


def test_resolve_path_pydantic_model():
  from pydantic import BaseModel

  class User(BaseModel):
    id: int
    name: str

  result = StepResult(data=User(id=1, name="Alice"))
  # path "data" resolves to the data attribute
  assert _resolve_path(result, "data").name == "Alice"
  assert _resolve_path(result, "data.id") == 1


def test_resolve_refs_basic():
  completed = {
    "s1": StepResult(data={"value": 42}),
    "s2": StepResult(data="hello"),
  }
  # source.data is {"value": 42}, so path "value" resolves to 42
  args = {"a": ref("s1", "value"), "b": 10}
  resolved = _resolve_refs(args, completed)
  assert resolved["a"] == 42
  assert resolved["b"] == 10


def test_resolve_refs_missing_step():
  with pytest.raises(ValueError, match="not found"):
    _resolve_refs({"x": ref("missing", "data")}, {})


# --------------------------------------------------------------------------- #
# PlanBuilder + ref() shorthand
# --------------------------------------------------------------------------- #

def test_plan_builder_chain_and_ref_shorthand():
  """ref() supports both explicit and shorthand calling conventions."""
  plan = (
    PlanBuilder(objective="Test")
    .step("fetch", "fetch_tool", url="https://example.com")
    .step("analyze", "analyze_tool", data=ref("fetch", "data"))
    .step("report", "report_tool", summary=ref("analyze.result"))
    .build()
  )

  assert len(plan.steps) == 3
  fetch_step = plan.get_step("fetch")
  analyze_step = plan.get_step("analyze")
  report_step = plan.get_step("report")

  assert fetch_step.depends_on == frozenset()
  assert analyze_step.depends_on == frozenset({"fetch"})
  assert report_step.depends_on == frozenset({"analyze"})

  # ref shorthand parsing
  r = ref("foo.bar.baz")
  assert r.step_id == "foo"
  assert r.path == "bar.baz"

  # ref with no dot returns raw step data (empty path)
  r2 = ref("foo")
  assert r2.step_id == "foo"
  assert r2.path == ""


def test_plan_builder_duplicate_id_raises():
  builder = PlanBuilder()
  builder.step("s1", "tool_a")
  with pytest.raises(ValueError, match="Duplicate"):
    builder.step("s1", "tool_b")


def test_plan_builder_callable_tool():
  @tool
  async def my_tool(x: int) -> int:
    return x

  plan = PlanBuilder().step("s1", my_tool, x=1).build()
  assert plan.get_step("s1").tool == "my_tool"


# --------------------------------------------------------------------------- #
# PlanExecutor (was DAGScheduler)
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_plan_executor_executes_plan():
  @tool
  async def add(ctx: RunContext, a: int, b: int) -> int:
    return a + b

  @tool
  async def mul(ctx: RunContext, a: int, b: int) -> int:
    return a * b

  agent = Agent(model="test", tools=[add, mul])
  session = Session(session_id="test", agent=agent, deps=None)
  store = MemoryCheckpointStore()

  plan = Plan(
    objective="Math",
    steps=(
      Step(id="s1", tool="add", args={"a": 1, "b": 2}),
      Step(id="s2", tool="mul", args={"a": 3, "b": 4}, depends_on=frozenset({"s1"})),
    ),
  )
  session.current_plan = plan

  from nonoka.core.hooks import Hooks
  class MockRunner:
    def __init__(self):
      self.checkpoint_store = store
      self.llm = None
      self.hooks = Hooks()

  runner = MockRunner()
  executor = PlanExecutor()
  result = await executor.execute(plan, session, runner)

  assert result.success is True
  assert session.status == SessionStatus.COMPLETED
  assert "s1" in session.completed_steps
  assert "s2" in session.completed_steps
  # Tool results are normalised to {"result": value, "has_more": False}
  assert session.completed_steps["s1"].data == {"result": 3, "has_more": False}
  assert session.completed_steps["s2"].data == {"result": 12, "has_more": False}


@pytest.mark.asyncio
async def test_plan_executor_ref_resolution():
  @tool
  async def concat(ctx: RunContext, prefix: str, suffix: str) -> str:
    return prefix + suffix

  agent = Agent(model="test", tools=[concat])
  session = Session(session_id="test", agent=agent, deps=None)
  store = MemoryCheckpointStore()

  plan = Plan(
    objective="Ref test",
    steps=(
      Step(id="s1", tool="concat", args={"prefix": "Hello, ", "suffix": "world"}),
      Step(id="s2", tool="concat", args={"prefix": ref("s1"), "suffix": "!"}),
    ),
  )
  session.current_plan = plan

  from nonoka.core.hooks import Hooks
  class MockRunner:
    def __init__(self):
      self.checkpoint_store = store
      self.llm = None
      self.hooks = Hooks()

  runner = MockRunner()
  executor = PlanExecutor()
  result = await executor.execute(plan, session, runner)

  assert result.success is True
  assert session.completed_steps["s2"].data == {"result": "Hello, world!", "has_more": False}


@pytest.mark.asyncio
async def test_plan_executor_step_retry_and_failure():
  call_count = 0

  @tool
  async def flaky(ctx: RunContext) -> str:
    nonlocal call_count
    call_count += 1
    if call_count < 3:
      raise TransientError("not yet")
    return "ok"

  from nonoka.core.types import RetryPolicy
  agent = Agent(model="test", tools=[flaky], default_retry=RetryPolicy(max_retries=2, backoff=0.01))
  session = Session(session_id="test", agent=agent, deps=None)
  store = MemoryCheckpointStore()

  plan = Plan(
    objective="Retry test",
    steps=(
      Step(id="s1", tool="flaky", retry=RetryPolicy(max_retries=2, backoff=0.01)),
    ),
  )
  session.current_plan = plan

  from nonoka.core.hooks import Hooks
  class MockRunner:
    def __init__(self):
      self.checkpoint_store = store
      self.llm = None
      self.hooks = Hooks()

  runner = MockRunner()
  executor = PlanExecutor()
  result = await executor.execute(plan, session, runner)

  assert result.success is True
  assert call_count == 3
  assert session.completed_steps["s1"].data == {"result": "ok", "has_more": False}


@pytest.mark.asyncio
async def test_plan_executor_step_timeout():
  @tool
  async def slow(ctx: RunContext) -> str:
    await asyncio.sleep(10)
    return "done"

  agent = Agent(model="test", tools=[slow])
  session = Session(session_id="test", agent=agent, deps=None)
  store = MemoryCheckpointStore()

  plan = Plan(
    objective="Timeout test",
    steps=(
      Step(id="s1", tool="slow", timeout=0.1),
    ),
  )
  session.current_plan = plan

  from nonoka.core.hooks import Hooks
  class MockRunner:
    def __init__(self):
      self.checkpoint_store = store
      self.llm = None
      self.hooks = Hooks()

  runner = MockRunner()
  executor = PlanExecutor()
  result = await executor.execute(plan, session, runner)

  assert result.success is False
  assert "s1" in session.failed_steps
  assert session.step_statuses["s1"] == StepStatus.FAILED


@pytest.mark.asyncio
async def test_plan_executor_skips_completed_on_resume():
  @tool
  async def identity(ctx: RunContext, x: int) -> int:
    return x

  agent = Agent(model="test", tools=[identity])
  session = Session(session_id="test", agent=agent, deps=None)
  store = MemoryCheckpointStore()

  plan = Plan(
    objective="Resume test",
    steps=(
      Step(id="s1", tool="identity", args={"x": 1}),
      Step(id="s2", tool="identity", args={"x": 2}),
    ),
  )
  session.current_plan = plan
  # Simulate checkpoint state: s1 already done
  session.completed_steps["s1"] = StepResult(data=100)
  session.step_statuses["s1"] = StepStatus.COMPLETED

  from nonoka.core.hooks import Hooks
  class MockRunner:
    def __init__(self):
      self.checkpoint_store = store
      self.llm = None
      self.hooks = Hooks()

  runner = MockRunner()
  executor = PlanExecutor()
  result = await executor.resume(plan, session, runner)

  assert result.success is True
  # s1 should keep its checkpointed result, s2 should execute
  assert session.completed_steps["s1"].data == 100
  assert session.completed_steps["s2"].data == {"result": 2, "has_more": False}


# --------------------------------------------------------------------------- #
# ReActAgent (was ConversationalScheduler)
# --------------------------------------------------------------------------- #

class MockLLM:
  """Mock LLM that alternates between tool_calls and final content."""

  def __init__(self, responses):
    self.responses = responses
    self.call_idx = 0

  async def chat(self, messages, tools=None, **kwargs):
    from nonoka.core.llm import LLMResponse
    resp = self.responses[self.call_idx]
    self.call_idx += 1
    return LLMResponse(**resp)


@pytest.mark.asyncio
async def test_react_agent_no_tool_calls():
  @tool
  async def noop(ctx: RunContext) -> str:
    return "noop"

  agent = Agent(model="test", tools=[noop], max_turns=3)
  session = Session(session_id="test", agent=agent, deps=None)
  store = MemoryCheckpointStore()

  mock_llm = MockLLM([
    {"content": "Final answer"},
  ])

  from nonoka.core.hooks import Hooks
  class MockRunner:
    def __init__(self):
      self.checkpoint_store = store
      self.llm = mock_llm
      self.hooks = Hooks()

  runner = MockRunner()
  paradigm = ReActAgent()
  result = await paradigm.run(session, runner, prompt="Hello")

  assert result.success is True
  assert result.data == "Final answer"
  assert session.status == SessionStatus.COMPLETED


@pytest.mark.asyncio
async def test_react_agent_tool_call_and_retry():
  call_count = 0

  @tool
  async def flaky_add(ctx: RunContext, a: int, b: int) -> int:
    nonlocal call_count
    call_count += 1
    if call_count == 1:
      raise TransientError("temp fail")
    return a + b

  agent = Agent(model="test", tools=[flaky_add], max_turns=3)
  session = Session(session_id="test", agent=agent, deps=None)
  store = MemoryCheckpointStore()

  mock_llm = MockLLM([
    {
      "content": None,
      "tool_calls": [
        {
          "id": "tc1",
          "function": {
            "name": "flaky_add",
            "arguments": '{"a": 1, "b": 2}',
          },
        }
      ],
    },
    {"content": "Done"},
  ])

  from nonoka.core.hooks import Hooks
  class MockRunner:
    def __init__(self):
      self.checkpoint_store = store
      self.llm = mock_llm
      self.hooks = Hooks()

  runner = MockRunner()
  paradigm = ReActAgent()
  result = await paradigm.run(session, runner, prompt="Add 1+2")

  assert result.success is True
  assert call_count == 2  # 1st failed, retried, 2nd succeeded


@pytest.mark.asyncio
async def test_react_agent_max_turns():
  agent = Agent(model="test", max_turns=2)
  session = Session(session_id="test", agent=agent, deps=None)
  store = MemoryCheckpointStore()

  # LLM always returns tool_calls so the loop never terminates naturally
  mock_llm = MockLLM([
    {
      "content": None,
      "tool_calls": [
        {
          "id": f"tc{i}",
          "function": {"name": "missing_tool", "arguments": "{}"},
        }
      ],
    }
    for i in range(3)
  ])

  from nonoka.core.hooks import Hooks
  class MockRunner:
    def __init__(self):
      self.checkpoint_store = store
      self.llm = mock_llm
      self.hooks = Hooks()

  runner = MockRunner()
  paradigm = ReActAgent()
  result = await paradigm.run(session, runner, prompt="Test")

  assert result.success is False
  assert "Max turns" in result.error
  assert session.status == SessionStatus.FAILED


# --------------------------------------------------------------------------- #
# Streaming interface
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_react_agent_run_stream_yields_content_and_final():
  """ReActAgent.run_stream should emit content_delta and a final event."""
  @tool
  async def no_op() -> str:
    return "done"

  agent = Agent(model="test", tools=[no_op], max_turns=3)
  session = Session(session_id="stream-1", agent=agent)

  from nonoka.core.hooks import Hooks
  runner = MagicMock()
  runner.checkpoint_store = MemoryCheckpointStore()
  runner.hooks = Hooks()

  llm_mock = MagicMock()

  async def fake_stream(*args, **kwargs):
    from nonoka.core.llm import LLMStreamChunk
    yield LLMStreamChunk(finish_reason="stop")

  llm_mock.chat_stream = fake_stream
  llm_mock.chat = AsyncMock(return_value=MagicMock(content="Final answer", tool_calls=None, usage={}))
  runner.llm = llm_mock

  react = ReActAgent()
  events = []
  async for event in react.run_stream(session, runner, prompt="hi"):
    events.append(event)

  if events[-1].type != "final":
    for ev in events:
      print("EVENT:", ev.type, ev.data)

  assert events[-1].type == "final"
  assert events[-1].data["success"] is True


# --------------------------------------------------------------------------- #
# Cancellation / Abort
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_react_agent_returns_cancelled_error_type():
  """ReActAgent should return RunResult with error_type='cancelled' when cancelled."""
  agent = Agent(model="test", tools=[], max_turns=1)
  session = Session(session_id="s3", agent=agent)

  session.cancel()

  from nonoka.core.hooks import Hooks
  runner = MagicMock()
  runner.checkpoint_store = MemoryCheckpointStore()
  runner.hooks = Hooks()

  react = ReActAgent()
  result = await react.run(session, runner, prompt="hello")

  assert result.success is False
  assert result.error_type == "cancelled"
  assert session.status == SessionStatus.CANCELLED


@pytest.mark.asyncio
async def test_plan_executor_returns_cancelled_error_type():
  """PlanExecutor should return RunResult with error_type='cancelled' when cancelled."""
  @tool
  async def dummy_tool(x: int) -> int:
    return x

  agent = Agent(model="test", tools=[dummy_tool])
  session = Session(session_id="s4", agent=agent)
  session.cancel()

  plan = (
    PlanBuilder()
    .step("s1", dummy_tool, x=1)
    .build()
  )

  from nonoka.core.hooks import Hooks
  runner = MagicMock()
  runner.checkpoint_store = MemoryCheckpointStore()
  runner.hooks = Hooks()

  executor = PlanExecutor()
  result = await executor.execute(plan, session, runner)

  assert result.success is False
  assert result.error_type == "cancelled"
  assert session.status == SessionStatus.CANCELLED


# --------------------------------------------------------------------------- #
# Error classification
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_react_agent_max_turns_error_type():
  """ReActAgent should return error_type='limit_exceeded' on max turns."""
  agent = Agent(model="test", tools=[], max_turns=0)
  session = Session(session_id="s5", agent=agent)

  from nonoka.core.hooks import Hooks
  runner = MagicMock()
  runner.checkpoint_store = MemoryCheckpointStore()
  runner.hooks = Hooks()

  react = ReActAgent()
  result = await react.run(session, runner, prompt="hello")

  assert result.success is False
  assert result.error_type == "limit_exceeded"


# --------------------------------------------------------------------------- #
# Plan layers caching
# --------------------------------------------------------------------------- #

def test_plan_layers_precomputed():
  """Plan.layers should be pre-computed at build time."""
  plan = (
    PlanBuilder()
    .step("a", "tool_a")
    .step("b", "tool_b")
    .step("c", "tool_c", depends_on={"a"})
    .build()
  )

  layers1 = plan.layers
  layers2 = plan.layers
  assert layers1 is layers2, "layers should be cached"
  assert layers1 == [["a", "b"], ["c"]]


def test_plan_topological_groups_alias():
  """topological_groups() should remain a backward-compatible alias."""
  plan = (
    PlanBuilder()
    .step("a", "tool_a")
    .step("b", "tool_b", depends_on={"a"})
    .build()
  )

  assert plan.topological_groups() == plan.layers


# --------------------------------------------------------------------------- #
# ReActAgent concurrency control
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_react_agent_respects_max_concurrency():
  """ReActAgent should use a Semaphore to limit concurrent tool calls."""
  concurrent_count = 0
  max_observed = 0

  @tool
  async def slow_tool(ctx: RunContext, delay: float) -> str:
    nonlocal concurrent_count, max_observed
    concurrent_count += 1
    max_observed = max(max_observed, concurrent_count)
    await asyncio.sleep(delay)
    concurrent_count -= 1
    return "done"

  agent = Agent(model="test", tools=[slow_tool])
  session = Session(session_id="s6", agent=agent)

  from nonoka.core.hooks import Hooks
  runner = MagicMock()
  runner.checkpoint_store = MemoryCheckpointStore()
  runner.hooks = Hooks()

  llm_mock = MagicMock()
  llm_mock.chat = AsyncMock(return_value=MagicMock(
    content="",
    tool_calls=[
      {"function": {"name": "slow_tool", "arguments": '{"delay": 0.1}'}},
      {"function": {"name": "slow_tool", "arguments": '{"delay": 0.1}'}},
      {"function": {"name": "slow_tool", "arguments": '{"delay": 0.1}'}},
      {"function": {"name": "slow_tool", "arguments": '{"delay": 0.1}'}},
      {"function": {"name": "slow_tool", "arguments": '{"delay": 0.1}'}},
    ],
  ))
  runner.llm = llm_mock

  react = ReActAgent(max_concurrency=2)
  result = await react.run(session, runner, prompt="run")

  assert max_observed <= 2, f"Expected max concurrency 2, got {max_observed}"


@pytest.mark.asyncio
async def test_react_agent_uses_agent_max_concurrency_default():
  """ReActAgent should fall back to agent.max_concurrency when not explicitly set."""
  agent = Agent(model="test", tools=[], max_concurrency=7)
  session = Session(session_id="s7", agent=agent)

  react = ReActAgent()
  assert react.max_concurrency is None
  assert agent.max_concurrency == 7


# --------------------------------------------------------------------------- #
# ToolEvaluator
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_tool_evaluator_uses_session():
  """ToolEvaluator should build RunContext from result.session correctly."""
  @tool
  async def validator(ctx: RunContext, data: Any) -> dict:
    assert ctx.session is not None
    return {"passed": True, "feedback": "ok"}

  agent = Agent(model="test", tools=[validator])
  session = Session(session_id="s8", agent=agent)

  evaluator = ToolEvaluator(validator, data_extractor=lambda r: r.data)

  mock_result = RunResult(success=True, data="test-data", session=session)
  eval_result = await evaluator.evaluate(mock_result)

  assert eval_result.passed is True
  assert eval_result.feedback == "ok"


@pytest.mark.asyncio
async def test_tool_evaluator_no_session():
  """ToolEvaluator should handle missing session gracefully."""
  @tool
  async def validator(ctx: RunContext, data: Any) -> dict:
    return {"passed": True}

  evaluator = ToolEvaluator(validator)

  mock_result = RunResult(success=True, data="test", session=None)
  eval_result = await evaluator.evaluate(mock_result)

  assert eval_result.passed is False
  assert "No session available" in eval_result.feedback


# --------------------------------------------------------------------------- #
# Ref list index resolution (Bug fix: P0)
# --------------------------------------------------------------------------- #

def test_resolve_path_list_index():
  """_resolve_path should support numeric string as list index: users.0"""
  data = {"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]}
  assert _resolve_path(data, "users.0") == {"id": 1, "name": "Alice"}
  assert _resolve_path(data, "users.1") == {"id": 2, "name": "Bob"}


def test_resolve_path_list_index_nested():
  """_resolve_path should support mixed dict+list path: users.0.name"""
  data = {"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]}
  assert _resolve_path(data, "users.0.name") == "Alice"
  assert _resolve_path(data, "users.1.name") == "Bob"


def test_resolve_path_list_index_out_of_bounds():
  """_resolve_path should return None for out-of-bounds list index."""
  data = {"users": [{"name": "Alice"}]}
  assert _resolve_path(data, "users.5") is None


def test_resolve_path_dict_key_that_looks_like_number():
  """Numeric string should still work as dict key when value is a dict."""
  data = {"0": "zero", "1": "one"}
  assert _resolve_path(data, "0") == "zero"
  assert _resolve_path(data, "1") == "one"


def test_resolve_path_list_of_primitives():
  """_resolve_path should work with lists of primitives."""
  data = {"scores": [100, 95, 87]}
  assert _resolve_path(data, "scores.0") == 100
  assert _resolve_path(data, "scores.2") == 87


def test_resolve_path_deeply_nested_list():
  """_resolve_path should handle deeply nested list access."""
  data = {"matrix": [[1, 2], [3, 4]]}
  assert _resolve_path(data, "matrix.0") == [1, 2]
  assert _resolve_path(data, "matrix.0.1") == 2
  assert _resolve_path(data, "matrix.1.0") == 3


def test_resolve_path_non_numeric_string_on_list():
  """Non-numeric string on a list should return None."""
  data = {"users": [{"name": "Alice"}]}
  assert _resolve_path(data, "users.name") is None


def test_resolve_refs_with_list_index():
  """Ref should resolve list indices in completed step data."""
  completed = {
    "fetch": StepResult(data={"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]}),
  }
  args = {"user": ref("fetch", "users.0")}
  resolved = _resolve_refs(args, completed)
  assert resolved["user"] == {"id": 1, "name": "Alice"}


def test_resolve_refs_with_list_index_nested():
  """Ref should resolve nested list+dict paths."""
  completed = {
    "fetch": StepResult(data={"users": [{"id": 1, "name": "Alice"}]}),
  }
  args = {"name": ref("fetch", "users.0.name")}
  resolved = _resolve_refs(args, completed)
  assert resolved["name"] == "Alice"


@pytest.mark.asyncio
async def test_plan_executor_ref_list_index():
  """PlanExecutor should resolve ref with list index in real execution."""
  @tool
  async def get_users(ctx: RunContext) -> dict:
    return {"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]}

  @tool
  async def greet_user(ctx: RunContext, name: str) -> str:
    return f"Hello, {name}!"

  agent = Agent(model="test", tools=[get_users, greet_user])
  session = Session(session_id="test", agent=agent, deps=None)
  store = MemoryCheckpointStore()

  plan = Plan(
    objective="Get user and greet",
    steps=(
      Step(id="fetch", tool="get_users"),
      Step(id="greet", tool="greet_user", args={"name": ref("fetch", "users.0.name")}, depends_on=frozenset({"fetch"})),
    ),
  )
  session.current_plan = plan

  from nonoka.core.hooks import Hooks
  class MockRunner:
    def __init__(self):
      self.checkpoint_store = store
      self.llm = None
      self.hooks = Hooks()

  runner = MockRunner()
  executor = PlanExecutor()
  result = await executor.execute(plan, session, runner)

  assert result.success is True
  assert session.completed_steps["greet"].data == {"result": "Hello, Alice!", "has_more": False}


@pytest.mark.asyncio
async def test_plan_executor_ref_list_index_second_element():
  """PlanExecutor should resolve ref accessing second list element."""
  @tool
  async def get_data(ctx: RunContext) -> dict:
    return {"items": ["first", "second", "third"]}

  @tool
  async def process(ctx: RunContext, item: str) -> str:
    return f"processed:{item}"

  agent = Agent(model="test", tools=[get_data, process])
  session = Session(session_id="test", agent=agent, deps=None)
  store = MemoryCheckpointStore()

  plan = Plan(
    objective="Get second item",
    steps=(
      Step(id="fetch", tool="get_data"),
      Step(id="proc", tool="process", args={"item": ref("fetch", "items.1")}, depends_on=frozenset({"fetch"})),
    ),
  )
  session.current_plan = plan

  from nonoka.core.hooks import Hooks
  class MockRunner:
    def __init__(self):
      self.checkpoint_store = store
      self.llm = None
      self.hooks = Hooks()

  runner = MockRunner()
  executor = PlanExecutor()
  result = await executor.execute(plan, session, runner)

  assert result.success is True
  assert session.completed_steps["proc"].data == {"result": "processed:second", "has_more": False}
