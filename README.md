# Nonoka

A production-grade, type-safe Python agent framework with deterministic orchestration, conversational execution, and first-class MCP integration.

## Features

- **Type-safe core** — Pydantic-validated schemas throughout; agents, tools, and plans are all strongly typed
- **Deterministic orchestration** — `Plan` + `Step` + `ref()` for explicit control flow, not just prompt-and-pray
- **Conversational execution** — `ReActAgent`, `ReflectiveAgent`, and `PlanExecutor` paradigms out of the box
- **First-class tools** — `@tool` decorator with automatic Pydantic schema generation
- **Prompt engineering** — `@prompt` decorator and `PromptTemplate` for composable, type-safe prompt construction
- **MCP ready** — built-in MCP (Model Context Protocol) support via `mcp`
- **Resilient execution** — structured error taxonomy (`TransientError`, `LogicError`, `SafetyError`, etc.) with configurable `RetryPolicy`
- **Observable hooks** — `Hooks` system for tracing, logging, and custom middleware
- **Multi-backend LLM** — powered by `litellm`, supporting OpenAI, Anthropic, DeepSeek, and 100+ providers

## Installation

```bash
pip install nonoka
```

Or with uv:

```bash
uv add nonoka
```

## Quick Start

```python
import asyncio
import nonoka

@nonoka.tool
async def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"Sunny in {city}!"

# Sync functions are also supported
@nonoka.tool
def get_time() -> str:
    """Get the current time."""
    return "It's noon."

async def main():
    agent = nonoka.Agent(
        name="weather-bot",
        tools=[get_weather],
    )
    result = await agent.run("What's the weather in Tokyo?")
    print(result.output)

asyncio.run(main())
```

## Plans & Orchestration

Explicit multi-step workflows with type-safe references:

```python
from nonoka import PlanBuilder, ref

plan = (
    PlanBuilder(objective="Research workflow")
    .step("research", search_tool, query="Latest AI breakthroughs")
    .step("summarize", summarize_tool, content=ref("research"))
    .build()
)

executor = nonoka.PlanExecutor(plan=plan)
result = await executor.run("Latest AI breakthroughs")
```

## Prompt Templates

Composable, type-safe prompts:

```python
from nonoka import prompt, PromptTemplate

@prompt
def translate(text: str, target: str = "Chinese") -> str:
    """Translate the following text to {target}:

    {text}
    """

# Or programmatically with Jinja2 syntax
tpl = PromptTemplate("Summarize this in {{style}}:\n{{content}}")
output = tpl.render(style="bullet points", content=long_text)
```

## ReAct Agent

```python
agent = nonoka.ReActAgent(tools=[search, calculator])
result = await agent.run("What is 42 * the current temperature in Paris?")
```

## Configuration

Nonoka supports three ways to configure agents: **declarative files** (YAML/JSON/TOML), **fluent builders**, and **direct code**.

### Declarative Config (YAML)

Write a `nonoka.yaml` and load it:

```yaml
# nonoka.yaml
agents:
  weather_assistant:
    model: gpt-4o
    system_prompt: "You are a weather assistant."
    max_turns: 10
    tools:
      - import: my_tools.weather:get_weather

  code_assistant:
    model: deepseek-chat
    system_prompt: "You are a coding assistant."

runner:
  checkpoint: memory
  memory: in_memory

defaults:
  model: deepseek-chat
  max_turns: 10
```

```python
from nonoka import Config

config = Config.load("nonoka.yaml")           # or Config.auto_find()
agent = config.agents["weather_assistant"].build()
runner = config.runner.build()
```

Single-agent shorthand (no `agents:` dict needed):

```yaml
agent:
  model: gpt-4o
  system_prompt: "You are helpful."
```

```python
agent = config.agent.build()
```

### Environment Variables in Config

Use `${VAR}` or `${VAR:-default}` in YAML values:

```yaml
agent:
  model: ${NONOKA_MODEL:-gpt-4o}
  system_prompt: ${NONOKA_PROMPT}
```

### Fluent Builder API

```python
from nonoka import AgentBuilder, tool

@tool
async def get_weather(city: str) -> str:
    return f"Sunny in {city}!"

agent = (
    AgentBuilder()
    .model("gpt-4o")
    .system_prompt("You are a weather assistant.")
    .tool(get_weather)
    .tool_by_import("my_tools.search:search_city")
    .max_turns(20)
    .retry(max_retries=5, backoff=1.5)
    .metadata(category="weather")
    .tag("production")
    .build()
)
```

### From Dict / YAML / JSON

```python
from nonoka import Agent

# From dict
agent = Agent.from_dict({
    "model": "gpt-4o",
    "tools": ["my_tools:get_weather"],
})

# From file
agent = Agent.from_yaml("agent.yaml")
agent = Agent.from_json("agent.json")
```

### Environment-driven Settings

Nonoka also integrates with `pydantic-settings` for framework-level config:

```python
from nonoka.core.config import settings

print(settings.default_model)   # from NONOKA_DEFAULT_MODEL env var
print(settings.openai_api_key)  # from NONOKA_OPENAI_API_KEY env var
```

## Requirements

- Python >= 3.10

## License

MIT
