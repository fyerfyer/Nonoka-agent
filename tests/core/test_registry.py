from nonoka.core.registry import ToolRegistry
from nonoka.core.tool import tool

def test_registry_double_wrapping():
  registry = ToolRegistry()
  
  @tool
  async def my_tool(a: int) -> int:
    return a
    
  registered_tool = registry.register(my_tool)
  
  assert registered_tool is my_tool
  assert registry._tools["my_tool"] is my_tool
  
  @registry.register(description="test desc")
  async def my_raw_func(b: int) -> int:
    return b
    
  assert "my_raw_func" in registry._tools
  assert registry._tools["my_raw_func"].description == "test desc"