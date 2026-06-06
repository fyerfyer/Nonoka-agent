"""Tests for the lightweight hooks / middleware mechanism."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from nonoka.core.hooks import Hooks, HookContext, _normalize_hook
from nonoka.core.agent import Agent
from nonoka.core.session import Session
from nonoka.core.tool import tool
from nonoka.core.llm import LLMMessage, LLMResponse
from nonoka.core.types import RunResult


# --------------------------------------------------------------------------- #
# Hook normalization
# --------------------------------------------------------------------------- #

def test_normalize_hook_wraps_sync_function():
  """Sync hooks should be automatically wrapped into async."""
  def sync_hook(ctx):
    ctx.extra["called"] = True

  normalized = _normalize_hook(sync_hook)
  assert normalized is not sync_hook
  assert normalized != sync_hook


@pytest.mark.asyncio
async def test_normalize_hook_preserves_async_function():
  """Async hooks should pass through unchanged."""
  async def async_hook(ctx):
    pass

  normalized = _normalize_hook(async_hook)
  assert normalized is async_hook


# --------------------------------------------------------------------------- #
# Hooks constructor + list style
# --------------------------------------------------------------------------- #

def test_hooks_empty_by_default():
  hooks = Hooks()
  assert hooks._store.get("on_session_start") == []
  assert hooks._store.get("on_llm_request") == []


def test_hooks_constructor_accepts_single_callable():
  async def my_hook(ctx):
    pass

  hooks = Hooks(on_session_start=my_hook)
  resolved = hooks._store.get("on_session_start")
  assert len(resolved) == 1


def test_hooks_constructor_accepts_list():
  async def hook1(ctx):
    pass

  async def hook2(ctx):
    pass

  hooks = Hooks(on_session_start=[hook1, hook2])
  resolved = hooks._store.get("on_session_start")
  assert len(resolved) == 2


# --------------------------------------------------------------------------- #
# Decorator style registration
# --------------------------------------------------------------------------- #

def test_hooks_decorator_registration():
  hooks = Hooks()

  @hooks.on_session_start
  async def hook1(ctx):
    pass

  @hooks.on_session_start
  async def hook2(ctx):
    pass

  resolved = hooks._store.get("on_session_start")
  assert len(resolved) == 2


def test_hooks_decorator_mixed_with_constructor():
  async def ctor_hook(ctx):
    pass

  hooks = Hooks(on_session_start=ctor_hook)

  @hooks.on_session_start
  async def deco_hook(ctx):
    pass

  resolved = hooks._store.get("on_session_start")
  assert len(resolved) == 2


# --------------------------------------------------------------------------- #
# Subclass style registration
# --------------------------------------------------------------------------- #

def test_hooks_subclass_method():
  class MyHooks(Hooks):
    async def on_session_start(self, ctx):
      ctx.extra["subclass_called"] = True

  hooks = MyHooks()
  resolved = hooks._store.get("on_session_start")
  assert len(resolved) == 1


def test_hooks_subclass_overridden_by_constructor():
  class MyHooks(Hooks):
    async def on_session_start(self, ctx):
      ctx.extra["subclass"] = True

  async def ctor_hook(ctx):
    ctx.extra["ctor"] = True

  hooks = MyHooks(on_session_start=ctor_hook)
  resolved = hooks._store.get("on_session_start")
  # Constructor arg takes priority
  assert len(resolved) == 1


# --------------------------------------------------------------------------- #
# Emit helpers (with mocked session)
# --------------------------------------------------------------------------- #

@pytest.fixture
def mock_hook_ctx():
  agent = Agent(model="test", tools=[])
  session = Session(session_id="sess-1", agent=agent, deps=None)
  runner = MagicMock()
  return HookContext(session=session, runner=runner)


@pytest.mark.asyncio
async def test_emit_session_start(mock_hook_ctx):
  calls = []

  async def hook(ctx):
    calls.append("start")

  hooks = Hooks(on_session_start=hook)
  await hooks.emit_session_start(mock_hook_ctx)
  assert calls == ["start"]


@pytest.mark.asyncio
async def test_emit_session_end(mock_hook_ctx):
  calls = []

  async def hook(ctx, result):
    calls.append(("end", result.success))

  hooks = Hooks(on_session_end=hook)
  result = RunResult(success=True)
  await hooks.emit_session_end(mock_hook_ctx, result)
  assert calls == [("end", True)]


@pytest.mark.asyncio
async def test_emit_llm_request(mock_hook_ctx):
  calls = []

  async def hook(ctx, messages, tools):
    calls.append((len(messages), tools))

  hooks = Hooks(on_llm_request=hook)
  messages = [LLMMessage(role="user", content="hello")]
  await hooks.emit_llm_request(mock_hook_ctx, messages, None)
  assert calls == [(1, None)]


@pytest.mark.asyncio
async def test_emit_llm_response(mock_hook_ctx):
  calls = []

  async def hook(ctx, response):
    calls.append(response.content)

  hooks = Hooks(on_llm_response=hook)
  response = LLMResponse(content="hi")
  await hooks.emit_llm_response(mock_hook_ctx, response)
  assert calls == ["hi"]


@pytest.mark.asyncio
async def test_emit_tool_start(mock_hook_ctx):
  calls = []

  async def hook(ctx, name, args):
    calls.append((name, args))

  hooks = Hooks(on_tool_start=hook)
  await hooks.emit_tool_start(mock_hook_ctx, "get_weather", {"city": "Beijing"})
  assert calls == [("get_weather", {"city": "Beijing"})]


@pytest.mark.asyncio
async def test_emit_tool_end(mock_hook_ctx):
  calls = []

  async def hook(ctx, name, args, result, error):
    calls.append((name, result, error))

  hooks = Hooks(on_tool_end=hook)
  await hooks.emit_tool_end(mock_hook_ctx, "calc", {"x": 1}, 42, None)
  assert calls == [("calc", 42, None)]


@pytest.mark.asyncio
async def test_emit_plan_start(mock_hook_ctx):
  calls = []

  async def hook(ctx):
    calls.append("plan_start")

  hooks = Hooks(on_plan_start=hook)
  await hooks.emit_plan_start(mock_hook_ctx)
  assert calls == ["plan_start"]


@pytest.mark.asyncio
async def test_emit_plan_step_start(mock_hook_ctx):
  calls = []

  async def hook(ctx, step_id, tool, args):
    calls.append((step_id, tool))

  hooks = Hooks(on_plan_step_start=hook)
  await hooks.emit_plan_step_start(mock_hook_ctx, "step1", "calc", {"x": 1})
  assert calls == [("step1", "calc")]


@pytest.mark.asyncio
async def test_emit_plan_step_end(mock_hook_ctx):
  calls = []

  async def hook(ctx, step_id, tool, result, error):
    calls.append((step_id, result, error))

  hooks = Hooks(on_plan_step_end=hook)
  await hooks.emit_plan_step_end(mock_hook_ctx, "step1", "calc", 42, None)
  assert calls == [("step1", 42, None)]


# --------------------------------------------------------------------------- #
# Multiple hooks execution order
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_multiple_hooks_execute_in_order(mock_hook_ctx):
  calls = []

  async def hook1(ctx):
    calls.append(1)

  async def hook2(ctx):
    calls.append(2)

  async def hook3(ctx):
    calls.append(3)

  hooks = Hooks(on_session_start=[hook1, hook2, hook3])
  await hooks.emit_session_start(mock_hook_ctx)
  assert calls == [1, 2, 3]


# --------------------------------------------------------------------------- #
# Hook exceptions propagate
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_hook_exception_propagates(mock_hook_ctx):
  async def bad_hook(ctx):
    raise RuntimeError("boom")

  hooks = Hooks(on_session_start=bad_hook)
  with pytest.raises(RuntimeError, match="boom"):
    await hooks.emit_session_start(mock_hook_ctx)


# --------------------------------------------------------------------------- #
# HookContext properties
# --------------------------------------------------------------------------- #

def test_hook_context_agent_property(mock_hook_ctx):
  assert mock_hook_ctx.agent is mock_hook_ctx.session.agent
