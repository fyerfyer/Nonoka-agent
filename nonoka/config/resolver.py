from __future__ import annotations

"""Tool import path resolution for declarative configuration."""

from typing import Any


class ConfigLoadError(Exception):
  """Raised when a configuration file cannot be loaded or is invalid."""


def resolve_tool_import(import_path: str) -> Any:
  """Resolve ``module.submodule:function_name`` to a callable.

  Examples:
    ``my_tools.weather:get_weather`` → imports ``my_tools.weather``,
    then retrieves ``get_weather`` attribute.
  """
  if ":" not in import_path:
    raise ConfigLoadError(
      f"Invalid tool import path '{import_path}'. Expected format: "
      "'module.submodule:function_name'"
    )
  module_path, attr_path = import_path.split(":", 1)
  try:
    import importlib
    module = importlib.import_module(module_path)
  except ImportError as exc:
    raise ConfigLoadError(
      f"Cannot import module '{module_path}' for tool '{import_path}': {exc}"
    ) from exc

  obj = module
  for attr in attr_path.split("."):
    obj = getattr(obj, attr, None)
    if obj is None:
      raise ConfigLoadError(
        f"Cannot find attribute '{attr}' in module '{module_path}' "
        f"for tool '{import_path}'"
      )
  return obj


def _resolve_tool_entry(entry: str | Any) -> Any:
  """Resolve a single tool entry to a Capability.

  Accepts:
    - ``str``: import path like ``"module:function"``
    - ``Capability``: used directly
    - ``callable``: wrapped with ``@tool``

  Returns:
    A ``Capability`` instance.
  """
  from nonoka.core.tool import tool as make_tool
  from nonoka.core.types import Capability

  if isinstance(entry, str):
    obj = resolve_tool_import(entry)
  elif isinstance(entry, Capability):
    return entry
  elif callable(entry):
    return make_tool(entry)
  else:
    raise TypeError(f"Invalid tool entry: {entry!r}")

  if isinstance(obj, Capability):
    return obj
  elif callable(obj):
    return make_tool(obj)
  else:
    raise TypeError(
      f"Tool import '{entry}' resolved to {type(obj).__name__}, "
      "expected a callable or Capability"
    )
