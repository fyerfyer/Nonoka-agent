from __future__ import annotations

import importlib
import importlib.util
import inspect
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from nonoka.core.types import Capability
from nonoka.core.registry import ToolRegistry
from nonoka.core.tool import tool as make_tool


class ToolListProxy(Sequence):
  """A proxy sequence that dynamically reflects a ToolRegistry's contents.

  This object is stored in ``Agent.tools`` so that tool lookups always
  see the *current* state of the registry — newly-registered tools are
  visible immediately and removed tools disappear on the next access.

  Usage::

      registry = ToolRegistry()
      agent = Agent(model="gpt-4o", tools=[registry])
      # agent.tools iterates over whatever is in *registry* right now
  """

  def __init__(self, static_tools: list[Capability], registries: list[ToolRegistry]):
    self._static = list(static_tools)
    self._registries = list(registries)

  def _snapshot(self) -> list[Capability]:
    """Build a fresh snapshot merging static tools + all registries."""
    result: list[Capability] = list(self._static)
    seen = {t.name for t in result}
    for reg in self._registries:
      for cap in reg.get_all():
        if cap.name not in seen:
          result.append(cap)
          seen.add(cap.name)
    return result

  def __iter__(self) -> Iterator[Capability]:
    return iter(self._snapshot())

  def __len__(self) -> int:
    return len(self._snapshot())

  def __getitem__(self, idx: int) -> Capability:  # type: ignore[override]
    return self._snapshot()[idx]

  def __contains__(self, item: object) -> bool:
    if isinstance(item, str):
      return any(t.name == item for t in self._snapshot())
    return item in self._snapshot()

  def __repr__(self) -> str:
    names = [t.name for t in self._snapshot()]
    return f"<ToolListProxy tools={names}>"


class PluginManager:
  """Runtime plugin manager for dynamic tool loading / unloading.

  Responsibilities:
  * Load Python functions from modules or file paths and wrap them as tools.
  * Unload tools by name (removing them from a shared registry).
  * Track which tools were loaded by which plugin for bulk unload.

  Usage::

      pm = PluginManager()
      agent = Agent(model="gpt-4o", tools=[pm.registry])

      # Load a tool from a module path
      pm.load_tool("my_plugins.weather:get_temperature")

      # Load a tool from a .py file
      pm.load_tool_from_file("/path/to/plugin.py")

      # Unload by name
      pm.unload_tool("get_temperature")

      # Bulk unload everything managed by this manager
      pm.unload_all()
  """

  def __init__(self, registry: ToolRegistry | None = None):
    self.registry = registry or ToolRegistry()
    # Track provenance: plugin source -> list of tool names loaded from it
    self._provenance: dict[str, list[str]] = {}

  # -- Single-tool operations ------------------------------------------

  def load_tool(
    self,
    import_path: str,
    *,
    description: str | None = None,
    default_timeout: float | None = None,
  ) -> Capability:
    """Load a tool from ``module.submodule:function_name`` string.

    The function is wrapped with the ``@tool`` decorator and added to
    ``self.registry``.
    """
    module_path, _, func_name = import_path.partition(":")
    if not func_name:
      raise ValueError(
        f"Invalid import path '{import_path}'. "
        "Expected format: module.submodule:function_name"
      )

    module = importlib.import_module(module_path)
    func = getattr(module, func_name, None)
    if func is None:
      raise ImportError(
        f"Function '{func_name}' not found in module '{module_path}'"
      )

    capability = make_tool(func, description=description, default_timeout=default_timeout)
    self.registry.add(capability)

    # Track provenance
    self._provenance.setdefault(import_path, []).append(capability.name)
    return capability

  def load_tool_from_file(
    self,
    file_path: str,
    func_name: str | None = None,
    *,
    description: str | None = None,
    default_timeout: float | None = None,
  ) -> Capability | list[Capability]:
    """Load tool(s) from a Python file path.

    If *func_name* is given, only that function is loaded.
    Otherwise **all** top-level functions/classes marked with ``@tool``
    or all async functions with type annotations are loaded.

    Returns the loaded capability (or a list when auto-discovering).
    """
    path = Path(file_path).expanduser().resolve()
    if not path.exists():
      raise FileNotFoundError(f"Plugin file not found: {file_path}")

    # Create a module from the file
    module_name = f"__nonoka_plugin__{path.stem}"
    if module_name in sys.modules:
      # Force reload if already loaded
      del sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
      raise ImportError(f"Cannot load spec from {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    if func_name:
      func = getattr(module, func_name, None)
      if func is None:
        raise ImportError(
          f"Function '{func_name}' not found in {file_path}"
        )
      capability = make_tool(func, description=description, default_timeout=default_timeout)
      self.registry.add(capability)
      self._provenance.setdefault(str(path), []).append(capability.name)
      return capability

    # Auto-discover: all callables that look like tools
    loaded: list[Capability] = []
    for attr_name in dir(module):
      if attr_name.startswith("_"):
        continue
      obj = getattr(module, attr_name)
      if not callable(obj) or inspect.isclass(obj):
        continue
      # Skip functions from stdlib / other modules
      if getattr(obj, "__module__", None) != module_name:
        continue
      capability = make_tool(obj, default_timeout=default_timeout)
      self.registry.add(capability)
      loaded.append(capability)

    self._provenance.setdefault(str(path), [c.name for c in loaded])
    return loaded if len(loaded) > 1 else (loaded[0] if loaded else [])

  def unload_tool(self, name: str) -> Capability | None:
    """Unload a tool by name from the managed registry.

    Returns the removed capability, or ``None`` if it did not exist.
    """
    # Remove from provenance tracking as well
    for source, names in list(self._provenance.items()):
      if name in names:
        names.remove(name)
        if not names:
          del self._provenance[source]
        break
    return self.registry.remove(name)

  def unload_all(self) -> list[str]:
    """Unload every tool from the managed registry.

    Returns the list of removed tool names.
    """
    all_names = self.registry.names()
    removed: list[str] = []
    for name in list(all_names):
      if self.registry.remove(name) is not None:
        removed.append(name)
    self._provenance.clear()
    return removed

  def reload_tool(self, import_path: str) -> Capability:
    """Unload then reload a tool by its import path."""
    # Try to find and remove the old one first
    _, _, func_name = import_path.partition(":")
    if func_name and self.registry.has(func_name):
      self.unload_tool(func_name)
    return self.load_tool(import_path)

  def reload_file(self, file_path: str, func_name: str | None = None) -> Capability | list[Capability]:
    """Unload then reload tools from a file."""
    path = str(Path(file_path).expanduser().resolve())
    if path in self._provenance:
      for name in list(self._provenance[path]):
        self.unload_tool(name)
    return self.load_tool_from_file(file_path, func_name)

  def loaded(self) -> list[str]:
    """Return all currently-loaded tool names."""
    return self.registry.names()

  def __repr__(self) -> str:
    return f"<PluginManager tools={self.loaded()}>"
