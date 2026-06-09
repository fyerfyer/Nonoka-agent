from __future__ import annotations

from pydantic import BaseModel
from pydantic import ValidationError
import inspect
import typing
from collections.abc import Callable, Coroutine
from typing import Any, get_type_hints

from pydantic import TypeAdapter, create_model

from nonoka.core.types import Capability, RetryPolicy
from nonoka.core.context import RunContext


def _is_run_context_type(hint: Any) -> bool:
    """Check whether *hint* resolves to ``RunContext`` (including generic forms).

    Handles:
    * ``RunContext`` (bare)
    * ``RunContext[AppDeps]`` (generic alias)
    * ``Annotated[RunContext[AppDeps], ...]`` (PEP 593 wrapper)
    * ``Optional[RunContext]`` (Union)
    """
    if hint is RunContext:
        return True

    origin = typing.get_origin(hint)
    if origin is RunContext:
        return True

    # Handle Annotated[..., ...]
    if hasattr(hint, "__metadata__") and hasattr(hint, "__args__"):
        return _is_run_context_type(hint.__args__[0])

    # Handle Union / Optional
    if origin is typing.Union and hasattr(hint, "__args__"):
        return any(_is_run_context_type(arg) for arg in hint.__args__)

    return False


class Tool(Capability):
  """
  Convert a normal Python function to a tool
  """
  def __init__(
    self,
    func: Callable[..., Any],
    description: str | None = None,
    default_retry: RetryPolicy | None = None,
    default_timeout: float | None = None,
  ):
    self._func = func
    self._is_async = inspect.iscoroutinefunction(func)
    self._name = func.__name__
    self._description = description or inspect.getdoc(func) or ""
    self.default_retry = default_retry or RetryPolicy()
    self.default_timeout = default_timeout
    self._sig = inspect.signature(func)
    self._type_hints = get_type_hints(func)

    # Inspect parameter of RunContext
    self._ctx_param_name = None
    for pname, _param in self._sig.parameters.items():
      hint = self._type_hints.get(pname)
      if (hint and _is_run_context_type(hint)) or pname == "ctx":
        self._ctx_param_name = pname
        break
    self._parameters_schema, self._params_model = self._build_parameters()

  @property
  def name(self) -> str:
    return self._name

  @property
  def description(self) -> str:
    return self._description

  @property
  def parameters(self) -> dict[str, Any]:
    return self._parameters_schema

  async def __call__(self, *args: Any, **kwargs: Any) -> Any:
    """Allow calling the tool directly like a normal async function.

    Positional arguments are bound according to the original function
    signature.  If the tool requires ``RunContext`` it must be passed as
    a keyword argument matching the parameter name (usually ``ctx``).
    """
    # Bind positional + keyword args using the original signature
    try:
      bound = self._sig.bind(*args, **kwargs)
      bound.apply_defaults()
    except TypeError as exc:
      raise TypeError(f"Tool '{self.name}' call failed: {exc}") from exc

    call_kwargs = dict(bound.arguments)

    # Validate with Pydantic (skip RunContext parameter)
    if self._params_model:
      validate_kwargs = {
        k: v for k, v in call_kwargs.items()
        if k != self._ctx_param_name
      }
      try:
        validated = self._params_model.model_validate(validate_kwargs)
        for k, v in validated.model_dump().items():
          call_kwargs[k] = v
      except ValidationError as e:
        raise ValueError(f"Tool '{self.name}' arguments validation failed:\n{e}")

    if self._is_async:
      return await self._func(**call_kwargs)
    return self._func(**call_kwargs)

  async def invoke(self, ctx: RunContext, arguments: dict[str, Any]) -> Any:
    from nonoka.core.tool_response import unwrap_tool_response

    # Convert and validate arguments
    if self._params_model:
      try:
        validated = self._params_model.model_validate(arguments)
        kwargs = validated.model_dump()
      except ValidationError as e:
        raise ValueError(f"Tool '{self.name}' arguments validation failed:\n{e}")
    else:
      kwargs = {}
    # Inject context
    if self._ctx_param_name:
      kwargs[self._ctx_param_name] = ctx

    if self._is_async:
      result = await self._func(**kwargs)
    else:
      result = self._func(**kwargs)

    # Normalise return value so the LLM always sees a consistent shape.
    # ToolResponse -> expanded dict; plain value -> {"result": value, "has_more": false}
    return unwrap_tool_response(result)

  def to_json_schema(self) -> dict[str, Any]:
    """OpenAI-compatible function schema for LLM tool-calling."""
    return {
      "type": "function",
      "function": {
        "name": self.name,
        "description": self.description,
        "parameters": self.parameters,
      },
    }

  def _build_parameters(self) -> tuple[dict[str, Any], type[BaseModel] | None]:
    """
    Use Pydantic to generate JSON Schema and return the validation model.
    """
    fields = {}
    for param_name, param in self._sig.parameters.items():
      if param_name == self._ctx_param_name:
        continue
      annotation = self._type_hints.get(param_name, Any)
      default = ... if param.default == inspect.Parameter.empty else param.default
      fields[param_name] = (annotation, default)
    if not fields:
      return {"type": "object", "properties": {}}, None
    # Create a Pydantic Model dynamically
    model = create_model(f"{self.name}_params", **fields)
    return model.model_json_schema(), model


def tool(
  func: Callable | None = None,
  *,
  description: str | None = None,
  default_retry: RetryPolicy | None = None,
  default_timeout: float | None = None,
):
  """
  Decorator for exposing tools
  Usage:
    @nonoka.core.tool
    async def my_func(ctx: RunContext, arg: str): ...

    @nonoka.core.tool(description="Custom description", default_timeout=10.0)
    async def my_func(arg: str): ...
  """
  def wrapper(f: Callable) -> Tool:
    return Tool(
      func=f,
      description=description,
      default_retry=default_retry,
      default_timeout=default_timeout,
    )

  if func is None:
    return wrapper
  return wrapper(func)
