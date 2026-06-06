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
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"Sunny in {city}!"

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
    PlanBuilder()
    .add_step("research", "Search for information")
    .add_step("summarize", "Summarize findings", depends_on=[ref("research")])
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

# Or programmatically
tpl = PromptTemplate("Summarize this in {style}:\n{content}")
output = tpl.render(style="bullet points", content=long_text)
```

## ReAct Agent

```python
agent = nonoka.ReActAgent(tools=[search, calculator])
result = await agent.run("What is 42 * the current temperature in Paris?")
```

## Configuration

Nonoka integrates with `pydantic-settings` for environment-driven config:

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str
    default_model: str = "gpt-4o"

    class Config:
        env_prefix = "NONOKA_"
```

## Requirements

- Python >= 3.10

## License

MIT
