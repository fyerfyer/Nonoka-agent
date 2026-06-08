#!/usr/bin/env python3
"""
Test script for tool hot-reload functionality.
Uses DeepSeek API via .env configuration.

Run: python test_hot_reload.py
"""
from __future__ import annotations

import asyncio
import os
import sys

# Ensure we use the local nonoka
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nonoka import Agent, Runner, PluginManager
from nonoka.core.tool import tool


# -- A simple calculator tool for testing hot reload ----------------------

@tool
def calculator(expression: str) -> str:
  """Evaluate a mathematical expression.

  Args:
    expression: A math expression like "2 + 3 * 4" or "sqrt(16)".
  """
  import math
  # Whitelist safe names
  safe = {"__builtins__": {}}
  safe.update({name: getattr(math, name) for name in dir(math) if not name.startswith("_")})
  safe.update({"abs": abs, "max": max, "min": min, "sum": sum, "pow": pow, "round": round})
  try:
    result = eval(expression, safe)
    return str(result)
  except Exception as e:
    return f"Error: {e}"


@tool
def get_time() -> str:
  """Return the current local time."""
  from datetime import datetime
  return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# -- Test flow ------------------------------------------------------------

async def main():
  pm = PluginManager()

  # Create agent with the plugin manager's registry (empty initially)
  agent = Agent(
    model="deepseek-chat",
    tools=[pm.registry],
    system_prompt="You are a helpful assistant. If a user asks for math, use the calculator tool. If asked for time, use the get_time tool.",
    max_turns=5,
  )
  runner = Runner()

  print("=" * 60)
  print("STEP 1: Agent starts with NO tools")
  print(f"  Registry contents: {pm.loaded()}")
  print("=" * 60)

  result1 = await runner.run_react(agent, "What is 123 + 456?", deps=None)
  print(f"  > User: What is 123 + 456?")
  print(f"  > Agent: {result1.data}")
  print()

  print("=" * 60)
  print("STEP 2: Hot-load 'calculator' tool")
  print("=" * 60)
  pm.registry.add(calculator)
  print(f"  Registry contents after load: {pm.loaded()}")
  print()

  result2 = await runner.run_react(agent, "What is 123 + 456?", deps=None)
  print(f"  > User: What is 123 + 456?")
  print(f"  > Agent: {result2.data}")
  print()

  print("=" * 60)
  print("STEP 3: Hot-load 'get_time' tool")
  print("=" * 60)
  pm.registry.add(get_time)
  print(f"  Registry contents after load: {pm.loaded()}")
  print()

  result3 = await runner.run_react(agent, "What time is it now?", deps=None)
  print(f"  > User: What time is it now?")
  print(f"  > Agent: {result3.data}")
  print()

  print("=" * 60)
  print("STEP 4: Hot-unload 'calculator' tool")
  print("=" * 60)
  removed = pm.unload_tool("calculator")
  print(f"  Removed: {removed.name if removed else 'None'}")
  print(f"  Registry contents after unload: {pm.loaded()}")
  print()

  result4 = await runner.run_react(agent, "What is 999 * 888?", deps=None)
  print(f"  > User: What is 999 * 888?")
  print(f"  > Agent: {result4.data}")
  print()

  print("=" * 60)
  print("STEP 5: Hot-unload ALL tools")
  print("=" * 60)
  removed_all = pm.unload_all()
  print(f"  Removed: {removed_all}")
  print(f"  Registry contents after unload all: {pm.loaded()}")
  print()

  result5 = await runner.run_react(agent, "What time is it now?", deps=None)
  print(f"  > User: What time is it now?")
  print(f"  > Agent: {result5.data}")
  print()

  print("=" * 60)
  print("All tests completed!")
  print("=" * 60)


if __name__ == "__main__":
  asyncio.run(main())
