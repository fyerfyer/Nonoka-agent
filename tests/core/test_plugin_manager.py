"""Tests for PluginManager and hot-reload helpers."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from nonoka import PluginManager, tool


class TestPluginManagerLoadFromFile:
  """Tests for PluginManager.load_tool_from_file."""

  def test_load_undecorated_function(self, tmp_path):
    plugin_file = tmp_path / "plugin.py"
    plugin_file.write_text(
      """
def greet(name: str) -> str:
  return f"Hello, {name}!"
"""
    )

    pm = PluginManager()
    cap = pm.load_tool_from_file(str(plugin_file), "greet")

    assert cap.name == "greet"
    assert "greet" in pm.registry.names()

  def test_load_already_decorated_tool(self, tmp_path):
    plugin_file = tmp_path / "plugin.py"
    plugin_file.write_text(
      """
from nonoka import tool

@tool
def greet(name: str) -> str:
  return f"Hello, {name}!"
"""
    )

    pm = PluginManager()
    cap = pm.load_tool_from_file(str(plugin_file), "greet")

    assert cap.name == "greet"
    assert "greet" in pm.registry.names()

  def test_auto_discover_decorated_tools(self, tmp_path):
    plugin_file = tmp_path / "plugin.py"
    plugin_file.write_text(
      """
from nonoka import tool

@tool
def hello() -> str:
  return "hello"

@tool
def world() -> str:
  return "world"

# This should not be picked up (private)
def _internal() -> str:
  return "internal"
"""
    )

    pm = PluginManager()
    result = pm.load_tool_from_file(str(plugin_file))

    assert isinstance(result, list)
    names = {cap.name for cap in result}
    assert names == {"hello", "world"}
    assert "_internal" not in pm.registry.names()

  def test_load_missing_function_raises(self, tmp_path):
    plugin_file = tmp_path / "plugin.py"
    plugin_file.write_text("def existing() -> str:\n  return 'ok'\n")

    pm = PluginManager()
    with pytest.raises(ImportError, match="missing_fn"):
      pm.load_tool_from_file(str(plugin_file), "missing_fn")

  def test_unload_tool(self, tmp_path):
    plugin_file = tmp_path / "plugin.py"
    plugin_file.write_text(
      """
def greet(name: str) -> str:
  return f"Hello, {name}!"
"""
    )

    pm = PluginManager()
    pm.load_tool_from_file(str(plugin_file), "greet")
    assert "greet" in pm.registry.names()

    removed = pm.unload_tool("greet")
    assert removed is not None
    assert removed.name == "greet"
    assert "greet" not in pm.registry.names()
