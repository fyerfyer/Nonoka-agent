import pytest
import asyncio
from nonoka.core.agent import Agent
from nonoka.core.tool import tool
from nonoka.core.context import RunContext
from nonoka.core.session import Session, SessionStatus, StepStatus, StepResult, StepFailure
from nonoka.core.plan import Plan, Step, PlanBuilder, ref, Ref
from nonoka.core.scheduler import (
  ConversationalScheduler,
  DAGScheduler,
  _resolve_refs,
  _resolve_path,
)
from nonoka.core.errors import ErrorPolicy, TransientError, LogicError
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
  args = {"a": Ref("s1", "value"), "b": 10}
  resolved = _resolve_refs(args, completed)
  assert resolved["a"] == 42
  assert resolved["b"] == 10


def test_resolve_refs_missing_step():
  with pytest.raises(ValueError, match="not found"):
    _resolve_refs({"x": Ref("missing", "data")}, {})


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
# DAGScheduler
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_dag_scheduler_executes_plan():
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

  class MockRunner:
    def __init__(self):
      self.checkpoint_store = store
      self.llm = None

  runner = MockRunner()
  scheduler = DAGScheduler()
  result = await scheduler.run_plan(session, runner)

  assert result.success is True
  assert session.status == SessionStatus.COMPLETED
  assert "s1" in session.completed_steps
  assert "s2" in session.completed_steps
  assert session.completed_steps["s1"].data == 3
  assert session.completed_steps["s2"].data == 12


@pytest.mark.asyncio
async def test_dag_scheduler_ref_resolution():
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

  class MockRunner:
    def __init__(self):
      self.checkpoint_store = store
      self.llm = None

  runner = MockRunner()
  scheduler = DAGScheduler()
  result = await scheduler.run_plan(session, runner)

  assert result.success is True
  assert session.completed_steps["s2"].data == "Hello, world!"


@pytest.mark.asyncio
async def test_dag_scheduler_step_retry_and_failure():
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

  class MockRunner:
    def __init__(self):
      self.checkpoint_store = store
      self.llm = None

  runner = MockRunner()
  scheduler = DAGScheduler()
  result = await scheduler.run_plan(session, runner)

  assert result.success is True
  assert call_count == 3
  assert session.completed_steps["s1"].data == "ok"


@pytest.mark.asyncio
async def test_dag_scheduler_step_timeout():
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

  class MockRunner:
    def __init__(self):
      self.checkpoint_store = store
      self.llm = None

  runner = MockRunner()
  scheduler = DAGScheduler()
  result = await scheduler.run_plan(session, runner)

  assert result.success is False
  assert "s1" in session.failed_steps
  assert session.step_statuses["s1"] == StepStatus.FAILED


@pytest.mark.asyncio
async def test_dag_scheduler_skips_completed_on_resume():
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

  class MockRunner:
    def __init__(self):
      self.checkpoint_store = store
      self.llm = None

  runner = MockRunner()
  scheduler = DAGScheduler()
  result = await scheduler.resume(session, runner)

  assert result.success is True
  # s1 should keep its checkpointed result, s2 should execute
  assert session.completed_steps["s1"].data == 100
  assert session.completed_steps["s2"].data == 2


# --------------------------------------------------------------------------- #
# ConversationalScheduler
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
async def test_conversational_scheduler_no_tool_calls():
  @tool
  async def noop(ctx: RunContext) -> str:
    return "noop"

  agent = Agent(model="test", tools=[noop], max_turns=3)
  session = Session(session_id="test", agent=agent, deps=None)
  store = MemoryCheckpointStore()

  mock_llm = MockLLM([
    {"content": "Final answer"},
  ])

  class MockRunner:
    def __init__(self):
      self.checkpoint_store = store
      self.llm = mock_llm

  runner = MockRunner()
  scheduler = ConversationalScheduler()
  result = await scheduler.run(session, runner, prompt="Hello")

  assert result.success is True
  assert result.data == "Final answer"
  assert session.status == SessionStatus.COMPLETED


@pytest.mark.asyncio
async def test_conversational_scheduler_tool_call_and_retry():
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

  class MockRunner:
    def __init__(self):
      self.checkpoint_store = store
      self.llm = mock_llm

  runner = MockRunner()
  scheduler = ConversationalScheduler()
  result = await scheduler.run(session, runner, prompt="Add 1+2")

  assert result.success is True
  assert call_count == 2  # 1st failed, retried, 2nd succeeded


@pytest.mark.asyncio
async def test_conversational_scheduler_max_turns():
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

  class MockRunner:
    def __init__(self):
      self.checkpoint_store = store
      self.llm = mock_llm

  runner = MockRunner()
  scheduler = ConversationalScheduler()
  result = await scheduler.run(session, runner, prompt="Test")

  assert result.success is False
  assert "Max turns" in result.error
  assert session.status == SessionStatus.FAILED