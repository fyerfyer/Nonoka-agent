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
        model="gpt-4o",
        tools=[get_weather, get_time],
    )
    runner = nonoka.Runner()          # execution coordinator
    result = await runner.run_react(agent, "What's the weather in Tokyo?", deps=None)
    print(result.data)                # result.data (not result.output)

asyncio.run(main())
```

> **Key concept:** `Agent` is a pure configuration object.  Execution is handled by `Runner`, which owns the LLM provider, checkpoint store, and memory backend.

## Plans & Orchestration

Explicit multi-step workflows with type-safe references, executed deterministically via `Runner.run_plan`:

```python
from nonoka import PlanBuilder, ref, Runner

plan = (
    PlanBuilder(objective="Research workflow")
    .step("research", search_tool, query="Latest AI breakthroughs")
    .step("summarize", summarize_tool, content=ref("research"))
    .build()
)

runner = Runner()
result = await runner.run_plan(agent, plan=plan, deps=None)
print(result.data)
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
from nonoka import Agent, tool, Runner

@tool
async def search(query: str) -> dict:
    ...

@tool
async def calculator(expr: str) -> float:
    ...

agent = Agent(model="gpt-4o", tools=[search, calculator])
runner = Runner()
result = await runner.run_react(agent, "What is 42 * the current temperature in Paris?", deps=None)
print(result.data)
```

## Tool Responses

Tools can return plain values or a `ToolResponse` to communicate pagination and metadata to the agent loop:

```python
from nonoka import ToolResponse, tool

@tool
async def search_web(ctx, query: str, cursor: str | None = None) -> ToolResponse:
    results, next_cursor = await _do_search(query, cursor)
    return ToolResponse(
        data={"results": results, "query": query},
        has_more=next_cursor is not None,
        next_cursor=next_cursor,
        suggested_next_step="Summarise the findings and stop searching."
        if len(results) >= 5 else "Refine query and search again.",
    )
```

## Gateway (IM Platform Integration)

`Gateway` standardizes requests from QQ, Telegram, Discord, etc. and routes them to Agents, then pushes Agent outputs back to the original platforms.

```python
from nonoka.ext.gateway.core import Gateway
from nonoka.ext.gateway.limiter import TokenBucketLimiter

runner = Runner()
gateway = Gateway(runner, limiter=TokenBucketLimiter(default_rate=1, default_burst=3))
gateway.register_adapter(TelegramAdapter(token="..."))
gateway.set_default_agent(agent)

await gateway.start()
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

# Runner backend configuration (defaults are SQLite persistent)
# Use "memory" / "disabled" for testing
runner:
  checkpoint: sqlite        # or "memory", "disabled"
  memory: sqlite            # or "in_memory", "disabled"

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
