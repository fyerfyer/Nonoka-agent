import os
import pytest

from dotenv import load_dotenv

from nonoka import Agent, Runner, Hooks, tool


load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")


@pytest.fixture
def real_runner():
  """Create a Runner with a real LLM provider."""
  if not API_KEY:
    pytest.skip("No OPENAI_API_KEY found in environment.")

  model_name = "deepseek-chat"
  if BASE_URL:
    model_name = f"openai/{model_name}"

  runner = Runner()
  from nonoka.core.llm import LiteLLMProvider
  provider = LiteLLMProvider(
    model=model_name,
    api_key=API_KEY,
    base_url=BASE_URL,
  )
  runner._llm_cache[model_name] = provider
  runner.llm = provider
  return runner


# --------------------------------------------------------------------------- #
# Integration: ReAct execution with hooks
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_react_hooks_with_real_llm(real_runner):
  """Verify hooks are triggered during ReAct execution with real LLM."""
  events = []

  hooks = Hooks()

  @hooks.on_session_start
  async def on_start(ctx):
    events.append(("session_start", ctx.session.session_id))

  @hooks.on_llm_request
  async def on_llm_req(ctx, messages, tools):
    events.append(("llm_request", len(messages)))

  @hooks.on_llm_response
  async def on_llm_resp(ctx, response):
    events.append(("llm_response", response.content[:20] if response.content else None))

  @hooks.on_session_end
  async def on_end(ctx, result):
    events.append(("session_end", result.success))

  runner = Runner(hooks=hooks)
  # Re-use the real provider
  runner._llm_cache = real_runner._llm_cache.copy()
  runner.llm = real_runner.llm

  agent = Agent(
    model="openai/deepseek-chat",
    tools=[],
    system_prompt="You are a helpful assistant. Be very concise.",
    max_turns=2,
  )

  result = await runner.run_react(agent, prompt="Say 'hooks work' and nothing else.", deps=None)

  print(f"\n[Hooks events]: {events}")
  assert ("session_start", result.session.session_id) in events
  assert any(e[0] == "llm_request" for e in events)
  assert any(e[0] == "llm_response" for e in events)
  assert any(e[0] == "session_end" and e[1] is True for e in events)
  assert result.success is True


@pytest.mark.asyncio
async def test_react_hooks_with_tool_and_real_llm(real_runner):
  """Verify tool hooks are triggered when LLM calls a tool."""
  events = []

  hooks = Hooks()

  @hooks.on_tool_start
  async def on_tool_start(ctx, name, args):
    events.append(("tool_start", name, args))

  @hooks.on_tool_end
  async def on_tool_end(ctx, name, args, result, error):
    events.append(("tool_end", name, result, error))

  @tool
  async def calculate(ctx, expression: str) -> float:
    """Evaluate a mathematical expression."""
    return float(eval(expression, {"__builtins__": {}}, {}))

  runner = Runner(hooks=hooks)
  runner._llm_cache = real_runner._llm_cache.copy()
  runner.llm = real_runner.llm

  agent = Agent(
    model="openai/deepseek-chat",
    tools=[calculate],
    system_prompt="You are a calculator. Use the calculate tool for math.",
    max_turns=3,
  )

  result = await runner.run_react(
    agent,
    prompt="Calculate 7 * 8. Only call the tool, no extra text.",
    deps=None,
  )

  print(f"\n[Tool hooks events]: {events}")
  print(f"\n[Result]: success={result.success}, data={result.data!r}")

  # If LLM called the tool, we should see tool_start and tool_end events
  if any(e[0] == "tool_start" for e in events):
    assert any(e[0] == "tool_end" and e[3] is None for e in events)
  else:
    pytest.skip("LLM did not call the tool — this is LLM behavior, not a code bug.")


@pytest.mark.asyncio
async def test_plan_hooks_with_real_llm(real_runner):
  """Verify plan hooks are triggered during PlanExecutor execution."""
  from nonoka.core.plan import PlanBuilder

  events = []

  hooks = Hooks()

  @hooks.on_plan_start
  async def on_plan_start(ctx):
    events.append("plan_start")

  @hooks.on_plan_step_start
  async def on_step_start(ctx, step_id, tool, args):
    events.append(("step_start", step_id))

  @hooks.on_plan_step_end
  async def on_step_end(ctx, step_id, tool, result, error):
    events.append(("step_end", step_id, result, error))

  @tool
  async def double_value(ctx, x: int) -> int:
    return x * 2

  runner = Runner(hooks=hooks)
  runner._llm_cache = real_runner._llm_cache.copy()
  runner.llm = real_runner.llm

  agent = Agent(
    model="openai/deepseek-chat",
    tools=[double_value],
  )

  plan = (
    PlanBuilder(objective="Double a number")
    .step("s1", double_value, x=21)
    .build()
  )

  result = await runner.run_plan(agent, plan=plan, deps=None)

  print(f"\n[Plan hooks events]: {events}")
  assert "plan_start" in events
  assert ("step_start", "s1") in events
  assert any(e[0] == "step_end" and e[1] == "s1" and e[2] == 42 for e in events)
  assert result.success is True
  assert result.data == 42


@pytest.mark.asyncio
async def test_sync_hook_auto_wrap(real_runner):
  """Sync hooks should work seamlessly with async execution."""
  events = []

  hooks = Hooks()

  # This is a SYNC hook — should be auto-wrapped
  @hooks.on_session_start
  def on_start(ctx):  # Note: no async!
    events.append("sync_start")

  runner = Runner(hooks=hooks)
  runner._llm_cache = real_runner._llm_cache.copy()
  runner.llm = real_runner.llm

  agent = Agent(
    model="openai/deepseek-chat",
    tools=[],
    max_turns=2,
  )

  result = await runner.run_react(
    agent,
    prompt="Say 'sync hooks work' and nothing else.",
    deps=None,
  )

  print(f"\n[Sync hook events]: {events}")
  assert "sync_start" in events
  assert result.success is True
