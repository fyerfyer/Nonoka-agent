from pydantic import BaseModel
from pydantic import ValidationError
import inspect
from collections.abc import Callable, Coroutine
from typing import Any, get_type_hints

from pydantic import TypeAdapter, create_model

from nonoka.core.types import Capability, RetryPolicy, RunContext

class Tool(Capability):
  """
  Convert a normal Python function to a tool
  """
  def __init__(
    self,
    func: Callable[..., Coroutine[Any, Any, Any]],
    description: str | None = None,
    default_retry: RetryPolicy | None = None,
    default_timeout: float | None = None,
  ):
    if not inspect.iscoroutinefunction(func):
      raise TypeError(f"Tool function must be an async function: {func.__name__}")
    self._func = func
    self._name = func.__name__
    self._description = description or inspect.getdoc(func) or ""
    self.default_retry = default_retry or RetryPolicy()
    self.default_timeout = default_timeout
    self._sig = inspect.signature(func)
    self._type_hints = get_type_hints(func)
    
    # Inspect parameter of RunContext  
    self._ctx_param_name = None
    for name, param in self._sig.parameters.items():
      hint = self._type_hints.get(name)
      # Check if the type hint is RunContext
      if (hint and getattr(hint, "__origin__", hint) is RunContext) or name == "ctx":
        self._ctx_param_name = name
        break
    self._parameters_schema, self._params_model = self._build_parameters()
    self._returns_schema = self._build_returns_schema()
    
  @property
  def name(self) -> str:
    return self._name
    
  @property
  def description(self) -> str:
    return self._description
    
  @property
  def parameters(self) -> dict[str, Any]:
    return self._parameters_schema

  @property
  def returns(self) -> dict[str, Any]:
    return self._returns_schema
  async def invoke(self, ctx: RunContext, arguments: dict[str, Any]) -> Any:
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
      
    return await self._func(**kwargs)
    
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
    
  def _build_returns_schema(self) -> dict[str, Any]:
    return_type = self._type_hints.get("return", Any)
    if return_type is type(None):
      return {}
    return TypeAdapter(return_type).json_schema()


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