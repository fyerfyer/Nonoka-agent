from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from nonoka.core.hooks import Hooks, HookContext
from nonoka.core.agent import Agent
from nonoka.core.session import Session


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture
def mock_hook_ctx():
  agent = Agent(model="test", tools=[])
  session = Session(session_id="sess-1", agent=agent, deps=None)
  runner = MagicMock()
  return HookContext(session=session, runner=runner)


# --------------------------------------------------------------------------- #
# Single intercept hook
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_intercept_hook_modifies_arguments(mock_hook_ctx):
  """An intercept hook should be able to change the arguments dict."""

  async def add_prefix(ctx, tool_name, arguments):
    arguments["prefixed"] = True
    return arguments

  hooks = Hooks(on_tool_start_intercept=add_prefix)
  result = await hooks.emit_tool_start_intercept(
    mock_hook_ctx, "my_tool", {"value": 42}
  )

  assert result["value"] == 42
  assert result["prefixed"] is True


@pytest.mark.asyncio
async def test_intercept_hook_no_change_returns_original(mock_hook_ctx):
  """An intercept hook that does not mutate args should return them unchanged."""

  async def read_only(ctx, tool_name, arguments):
    # Inspect but don't modify
    _ = arguments.copy()
    return arguments

  hooks = Hooks(on_tool_start_intercept=read_only)
  original = {"value": 42}
  result = await hooks.emit_tool_start_intercept(
    mock_hook_ctx, "my_tool", original
  )

  assert result is original
  assert result == {"value": 42}


@pytest.mark.asyncio
async def test_intercept_hook_exception_propagates(mock_hook_ctx):
  """When an intercept hook raises, the exception should propagate upward
  and abort the tool call."""

  async def bad_hook(ctx, tool_name, arguments):
    raise ValueError("intercept boom")

  hooks = Hooks(on_tool_start_intercept=bad_hook)
  with pytest.raises(ValueError, match="intercept boom"):
    await hooks.emit_tool_start_intercept(
      mock_hook_ctx, "my_tool", {"value": 1}
    )


@pytest.mark.asyncio
async def test_intercept_hook_returns_non_dict_raises(mock_hook_ctx):
  """Intercept hooks must return a dict; anything else raises TypeError."""

  async def bad_hook(ctx, tool_name, arguments):
    return "not a dict"

  hooks = Hooks(on_tool_start_intercept=bad_hook)
  with pytest.raises(TypeError, match="must return a dict"):
    await hooks.emit_tool_start_intercept(
      mock_hook_ctx, "my_tool", {"value": 1}
    )


# --------------------------------------------------------------------------- #
# Multiple intercept hooks (composition)
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_multiple_intercept_hooks_chain(mock_hook_ctx):
  """Multiple intercept hooks should execute in order, each receiving
  the return value of the previous one."""

  calls = []

  async def hook1(ctx, tool_name, arguments):
    calls.append(("hook1", dict(arguments)))
    arguments["step1"] = True
    return arguments

  async def hook2(ctx, tool_name, arguments):
    calls.append(("hook2", dict(arguments)))
    arguments["step2"] = True
    return arguments

  async def hook3(ctx, tool_name, arguments):
    calls.append(("hook3", dict(arguments)))
    arguments["step3"] = True
    return arguments

  hooks = Hooks(on_tool_start_intercept=[hook1, hook2, hook3])
  result = await hooks.emit_tool_start_intercept(
    mock_hook_ctx, "my_tool", {"base": 0}
  )

  assert result == {"base": 0, "step1": True, "step2": True, "step3": True}
  # hook2 should see step1, hook3 should see step1+step2
  assert calls[1][1].get("step1") is True
  assert calls[2][1].get("step2") is True


@pytest.mark.asyncio
async def test_intercept_hooks_mixed_with_notification_hooks(mock_hook_ctx):
  """Intercept hooks and notification hooks should coexist independently."""

  notify_calls = []
  intercept_calls = []

  async def notify_hook(ctx, name, args):
    notify_calls.append(("notify", name, dict(args)))

  async def intercept_hook(ctx, name, args):
    intercept_calls.append(("intercept", name, dict(args)))
    args["modified"] = True
    return args

  hooks = Hooks(
    on_tool_start=notify_hook,
    on_tool_start_intercept=intercept_hook,
  )

  # Emit both hooks
  await hooks.emit_tool_start(mock_hook_ctx, "my_tool", {"value": 1})
  result = await hooks.emit_tool_start_intercept(
    mock_hook_ctx, "my_tool", {"value": 1}
  )

  # Notification hook should see original args (it runs before intercept)
  assert notify_calls == [("notify", "my_tool", {"value": 1})]
  # Intercept hook should modify args
  assert intercept_calls == [("intercept", "my_tool", {"value": 1})]
  assert result == {"value": 1, "modified": True}


# --------------------------------------------------------------------------- #
# No intercept hooks registered
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_no_intercept_hooks_returns_arguments_unchanged(mock_hook_ctx):
  """When no intercept hooks are registered, arguments pass through."""

  hooks = Hooks()
  original = {"value": 42}
  result = await hooks.emit_tool_start_intercept(
    mock_hook_ctx, "my_tool", original
  )

  assert result is original


# --------------------------------------------------------------------------- #
# Decorator registration
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_intercept_decorator_registration(mock_hook_ctx):
  """Intercept hooks should work with decorator registration."""

  hooks = Hooks()

  @hooks.on_tool_start_intercept
  async def add_flag(ctx, tool_name, arguments):
    arguments["flag"] = True
    return arguments

  result = await hooks.emit_tool_start_intercept(
    mock_hook_ctx, "my_tool", {"value": 1}
  )

  assert result["flag"] is True


# --------------------------------------------------------------------------- #
# Subclass style registration
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_intercept_subclass_method(mock_hook_ctx):
  """Subclasses of Hooks should be able to override on_tool_start_intercept."""

  class MyHooks(Hooks):
    async def on_tool_start_intercept(self, ctx, tool_name, arguments):
      arguments["from_subclass"] = True
      return arguments

  hooks = MyHooks()
  result = await hooks.emit_tool_start_intercept(
    mock_hook_ctx, "my_tool", {"value": 1}
  )

  assert result["from_subclass"] is True


@pytest.mark.asyncio
async def test_intercept_subclass_overridden_by_constructor(mock_hook_ctx):
  """Constructor arg for intercept should override subclass method."""

  class MyHooks(Hooks):
    async def on_tool_start_intercept(self, ctx, tool_name, arguments):
      arguments["subclass"] = True
      return arguments

  async def ctor_hook(ctx, tool_name, arguments):
    arguments["ctor"] = True
    return arguments

  hooks = MyHooks(on_tool_start_intercept=ctor_hook)
  result = await hooks.emit_tool_start_intercept(
    mock_hook_ctx, "my_tool", {"value": 1}
  )

  assert "ctor" in result
  assert "subclass" not in result
