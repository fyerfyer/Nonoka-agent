# Nonoka

[English](README.md) | 简体中文

一个生产级、类型安全的 Python Agent 框架，提供确定性编排、对话式执行和一流的 MCP 集成。

## 特性

- **类型安全的核心** —— 全链路 Pydantic 校验的 schema；agent、工具和 plan 都是强类型的
- **确定性编排** —— 通过 `Plan` + `Step` + `ref()` 实现显式的控制流，而不是仅靠 prompt 碰运气
- **对话式执行** —— 开箱即用的 `ReActAgent`、`ReflectiveAgent` 和 `PlanExecutor` 范式
- **一流的工具支持** —— `@tool` 装饰器自动生成 Pydantic schema
- **Prompt 工程** —— `@prompt` 装饰器和 `PromptTemplate`，用于可组合、类型安全的 prompt 构建
- **MCP 就绪** —— 内置 MCP（Model Context Protocol）生命周期管理器（`MCPManager`）和客户端（`MCPClient`）
- **懒加载技能** —— 无需膨胀 system prompt 即可发现和注册技能；通过 `load_skill` 工具按需加载完整指引
- **外部能力** —— 使用 `ExternalCapability` 和 `resume_external_tools()` 将工具执行委托给宿主/前端（例如 OpenCode）
- **弹性执行** —— 结构化的错误分类（`TransientError`、`LogicError`、`SafetyError` 等），配合可配置的 `RetryPolicy`
- **可观测 Hook** —— 用于追踪、日志和自定义中间件的 `Hooks` 系统
- **多后端 LLM** —— 基于 `litellm`，支持 OpenAI、Anthropic、DeepSeek 等 100+ 提供商

## 安装

```bash
pip install nonoka
```

或者使用 uv：

```bash
uv add nonoka
```

## 快速开始

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

> **核心概念：** `Agent` 是一个纯配置对象。执行由 `Runner` 负责，它持有 LLM provider、checkpoint 存储和 memory 后端。

## Plan 与编排

显式的多步骤工作流，带有类型安全的引用，通过 `Runner.run_plan` 确定性地执行：

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

## Prompt 模板

可组合、类型安全的 prompt：

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

## 工具响应

工具可以返回普通值，也可以返回 `ToolResponse`，以便向 agent 循环传递分页和元数据信息：

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

## 有状态工具与执行 trace

工具可以声明执行语义。显式的只读操作可以并发执行；
有状态的、会产生变更的、互斥的以及未知能力的操作会按照确定性的源码顺序串行执行。

```python
from nonoka import ToolExecution, tool

@tool(execution=ToolExecution(stateful_action=True, mutates_workspace=True))
async def run_terminal(command: str) -> str:
    ...

@tool(execution=ToolExecution(read_only=True, pagination=True))
async def read_log(cursor: str | None = None) -> str:
    ...
```

每个 `RunResult` 都携带一个有界且已脱敏（隐去凭证）的 `trace`。它包含
LLM 请求/响应的用量、工具耗时与结果、验证器（verifier）结果以及
最终的终止原因，适合作为 benchmark 产物使用且不会泄露 API key。

```python
result = await Runner().run_react(agent, "Inspect and fix the service", deps=None)
print(result.trace["termination"])
```

## 生产级可观测性

`Runner` 可以持久化已脱敏的 prompt、响应、工具 I/O、错误、token
用量以及 LiteLLM 成本估算。当配置了 SDK tracer provider 时，会为运行、
模型请求和工具调用发出 OpenTelemetry span。

```python
from nonoka import ObservabilityPipeline, Runner, SQLiteEventStore

pipeline = ObservabilityPipeline(
    SQLiteEventStore(".nonoka/events.db"),
    exporters=[my_exporter],  # Langfuse, OTLP, or another TelemetryExporter
)
runner = Runner(observability=pipeline)
```

Exporter 是可选的，并且是 best-effort 的，因此遥测后端故障不会
中断 agent 的执行。

## ASGI 服务与安全策略

带鉴权的 FastAPI 服务暴露 `/run`、流式 `/chat`、`/tasks`、
`/health` 以及兼容 Prometheus 的 `/metrics` 端点：

```bash
export NONOKA_API_TOKEN="replace-with-a-long-random-token"
uvicorn nonoka.server.app:create_app --factory --host 0.0.0.0 --port 8000
```

宿主也可以在执行工具之前复用文件系统和命令检查：

```python
from pathlib import Path
from nonoka import SafetyPolicy

policy = SafetyPolicy(allowed_roots=[Path.cwd()])
policy.check_path("src/app.py")
decision = policy.check_command("pytest -q")  # "allow" or "approval"
```

## 可选的循环扩展

默认循环保留了保守的工具调度器和进度守卫。
可选扩展可以在定义良好的位置加入有界反馈，而不改变
工具调用、并发或运行预算。它们的决策也会被
记录在 `result.trace["extensions"]` 中。

```python
from nonoka import Agent, Runner
from nonoka.ext.coding import VerifierRepairExtension

# evaluator implements: async evaluate(RunResult) -> EvaluationResult
agent = Agent(
    model="gpt-4o",
    tools=[...],
    extensions=[VerifierRepairExtension(evaluator, max_repairs=2)],
)
result = await Runner().run_react(agent, "Implement and verify the fix", deps=None)
```

`VerifierRepairExtension` 仅在确定性 verifier 失败后才请求再来一轮普通的 ReAct 回合。
`ResponseGroundingExtension` 同样可以基于工具建立的状态来校验最终的自然语言结论。
使用 `CodingWorkflow`（或 `CodeStrategyRouter`）可以根据调用方已知的任务能力
在 `direct`、`tool_assisted` 或 `verified_repair` 之间进行选择。
默认策略刻意保持保守：独立代码片段走 direct，工作区任务走 tool-assisted，
而 repair 则需要同时具备工作区和一个确定性的 evaluator。
`TerminalCodingWorkflow` 还要求调用方显式提供 `verify_command`；
它绝不会从 prompt 中猜测测试命令。`TerminalCommandEvaluator` 可以包装这个
已批准的命令和一个由调用方持有的终端执行器，为有界的 repair 扩展返回
结构化的测试失败信息。

## Gateway（IM 平台集成）

`Gateway` 对来自 QQ、Telegram、Discord 等平台的请求进行标准化并路由到 Agent，然后将 Agent 的输出推送回原始平台。

```python
from nonoka.ext.gateway.core import Gateway
from nonoka.ext.gateway.limiter import TokenBucketLimiter

runner = Runner()
gateway = Gateway(runner, limiter=TokenBucketLimiter(default_rate=1, default_burst=3))
gateway.register_adapter(TelegramAdapter(token="..."))
gateway.set_default_agent(agent)

await gateway.start()
```

## 配置

Nonoka 支持三种配置 agent 的方式：**声明式文件**（YAML/JSON/TOML）、**流式构建器**和**直接写代码**。

### 声明式配置（YAML）

编写一个 `nonoka.yaml` 并加载它：

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
    model: deepseek/deepseek-v4-pro
    system_prompt: "You are a coding assistant."

# Runner backend configuration (defaults are SQLite persistent)
# Use "memory" / "disabled" for testing
runner:
  checkpoint: sqlite        # or "memory", "disabled"
  memory: sqlite            # or "in_memory", "disabled"

defaults:
  model: deepseek/deepseek-v4-pro
  max_turns: 10
```

```python
from nonoka import Config

config = Config.load("nonoka.yaml")           # or Config.auto_find()
agent = config.agents["weather_assistant"].build()
runner = config.runner.build()
```

单 agent 简写形式（不需要 `agents:` 字典）：

```yaml
agent:
  model: gpt-4o
  system_prompt: "You are helpful."
```

```python
agent = config.agent.build()
```

### 配置中的环境变量

在 YAML 值中使用 `${VAR}` 或 `${VAR:-default}`：

```yaml
agent:
  model: ${NONOKA_MODEL:-gpt-4o}
  system_prompt: ${NONOKA_PROMPT}
```

### 流式构建器 API

```python
from nonoka import AgentBuilder, ToolRegistry, tool

@tool
async def get_weather(city: str) -> str:
    return f"Sunny in {city}!"

registry = ToolRegistry()

@registry.register
async def search_city(name: str) -> str:
    return f"Found {name}"

agent = (
    AgentBuilder()
    .model("gpt-4o")
    .system_prompt("You are a weather assistant.")
    .tool(get_weather)
    .tool_registry(registry)                 # add a whole registry
    .tool_by_import("my_tools.search:search_city")
    .max_turns(20)
    .retry(max_retries=5, backoff=1.5)
    .metadata(category="weather")
    .tag("production")
    .build()
)
```

你也可以直接把 `ToolRegistry` 传给 `.tools()`：

```python
agent = AgentBuilder().model("gpt-4o").tools(registry).build()
```

### 技能

在构建器中直接应用预打包的技能：

```python
from nonoka import AgentBuilder, Skill

skill = Skill.from_file("skills/code-review.md")

agent = (
    AgentBuilder()
    .model("gpt-4o")
    .system_prompt("You are a senior engineer.")
    .skill(skill)
    # or .skills(skill_a, skill_b)
    .build()
)
```

### 懒加载技能

对于拥有大量技能的项目，把每个技能都急切地合并进 system prompt 会撑爆上下文长度。使用 `SkillRegistry` 只暴露名称和描述，让模型在需要完整指引时自行调用 `load_skill`：

```python
from nonoka import AgentBuilder, SkillRegistry, load_skill

registry = SkillRegistry(enabled=["code-review", "nextjs-best-practices"])

agent = (
    AgentBuilder()
    .model("gpt-4o")
    .skill_manager(registry)
    .tool(load_skill)
    .build()
)
```

`load_skill` 工具会以 system message 的形式，将所选技能的 `system_prompt` 和 `activation_prompt` 注入对话中。

### MCP 服务器

通过 Model Context Protocol（MCP）连接外部工具和资源。nonoka-agent 提供了内置的 `MCPManager`，负责服务器生命周期（启动、健康检查、重启、关闭），并将发现的工具以普通 `Capability` 对象的形式暴露出来：

```python
from nonoka import AgentBuilder, Runner
from nonoka.ext.mcp import MCPManager, MCPServerConfig

manager = MCPManager()

configs = {
    "filesystem": MCPServerConfig(
        transport="stdio",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/home/user/docs"],
    ),
}

async def main():
    tools = await manager.start_all(configs)

    agent = (
        AgentBuilder()
        .model("gpt-4o")
        .system_prompt("Use the filesystem tools when needed.")
        # Register MCP tools individually (or merge them into a ToolRegistry)
        .tools(*[cap for _, cap in tools])
        .build()
    )

    runner = Runner()
    result = await runner.run_react(agent, "List the files in /home/user/docs")
    print(result.data)

    await manager.stop_all()
```

`MCPManager` 支持 stdio 和 sse 传输、并行启动、周期性健康检查以及指数退避重启。

### 外部能力

一些宿主（例如 OpenCode）希望自己掌控工具执行和人工审批环节。nonoka-agent 通过 `ExternalCapability` 支持这一点：框架注册工具 schema 并发出工具调用，但执行被委托给宿主。当宿主返回结果后，会话通过 `Runner.resume_external_tools()` 恢复。

```python
from nonoka import AgentBuilder, Runner, ExternalCapability, ToolExecution

cap = ExternalCapability(
    name="bash",
    description="Run a shell command.",
    parameters={
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
    execution=ToolExecution(stateful_action=True, mutates_workspace=True),
)

agent = AgentBuilder().model("gpt-4o").tool(cap).build()
runner = Runner()

# In the caller (e.g. nonoka-cli bridge):
# 1. Run until ExternalToolExecutionRequiredError is raised.
# 2. Forward the tool call to the external host.
# 3. Resume with the host's result. Workspace-mutating tools include a host
#    receipt and before/after workspace attestation.
async for event in runner.resume_external_tools(
    agent,
    deps=None,
    session_id="session-123",
    results={"call_abc": {
        "result": "command completed",
        "exit_code": 0,
        "elapsed_seconds": 0.14,
        "host": "my-terminal-host",
        "workspace": {
            "root": "/workspace",
            "before_digest": "...",
            "after_digest": "...",
            "created": ["solution.py"],
        },
    }},
):
    print(event)
```

`ExternalCapability` 携带 `external=True`，因此 ReAct 循环会暂停，而不是在本地调用该工具。这让 nonoka 专注于决策，而宿主负责执行、权限和 TUI 渲染。声明了 `ToolExecution(mutates_workspace=True)` 的能力会拒绝缺少上述回执（receipt）的恢复请求；该回执会被记录在已脱敏的 trace 中。它让跨进程的信任边界可审计，但并不会把一个不受信任的宿主变成沙箱。

#### 部分观测回退

外部宿主可以显式地将回执标记为 `completeness="partial"`：例如，它只能返回大型搜索结果的预览。宿主可以注册一个本地的只读能力作为**声明式观测回退**，使下一轮模型回合也能从兼容的本地操作中获得有界证据。

```python
from nonoka import tool, ToolExecution

@tool(description="Return small evidence snippets from a bounded local scope.",
      execution=ToolExecution(read_only=True))
async def bounded_probe(ctx, query: str, scope: str, limit: int = 20):
    ...

bounded_probe.metadata = {
    "kind": "observation_fallback",
    "fallback": {
        "on_partial_external": True,
        # fallback argument -> source external-call argument
        "argument_map": {"query": "query", "scope": "directory"},
        "defaults": {"limit": 20},
    },
}
```

当收到部分外部回执时，Nonoka 仅会在每个映射的源参数都存在且回退是只读的情况下，才选择一个已注册的声明。它执行该本地能力一次，并将其结构化结果附加到部分观测中，然后再恢复模型。框架不会按宿主名、外部工具名、任务名、路径或内容模式进行匹配；这些语义完全保留在能力声明中。缺少映射、非只读能力，或完整/未知的回执，都会直接跳过回退。

### 评估策略与外部 benchmark

框架代码任务默认采用直接生成。要对三种显式策略进行可复现的配对比较，请使用版本化的 complex MBPP 切片：

```bash
python -m nonoka.ext.eval compare --dataset mbpp-complex-v1 --model <model> --trials 3
```

Terminal-Bench 2 使用 Harbor 作为主要的官方 runner。Harbor 负责
Docker 生命周期和权威的任务产物，而 Nonoka 将其 trace 导出为 ATIF：

```bash
python -m nonoka.ext.eval external run --benchmark terminal-bench \
  --model <model> --task-id sanitize-git-repo --task-id configure-git-webserver
```

对于契约明确要求进行工作区编辑的任务，可以选择启用终端进度提醒，
而不是全局开启它。终端输出在进入模型上下文之前也会做有界截断；
这两个控制项都可以通过 Harbor agent kwargs 调整：

```bash
python -m nonoka.ext.eval external run --benchmark terminal-bench --model <model> \
  --task-id sanitize-git-repo --agent-kwarg requires_workspace_mutation=true \
  --agent-kwarg max_exploration_turns=3 --agent-kwarg max_terminal_output_chars=12000
```

导出的 ATIF 轨迹保留了逐回合的工具归属、有界的终端观测、用量、扩展决策和终止元数据。

将 `NONOKA_HARBOR_BIN` 设置为专用 Harbor 环境的可执行文件，并在开始真实 benchmark 之前运行 `python -m nonoka.ext.eval doctor`。
评估门禁刻意设计为两阶段：先运行确定性的 core/eval 适配器测试，
只有在 doctor 检查报告依赖就绪之后，才运行隔离的官方 harness。
`terminal-bench-legacy` 仅保留用于历史上的 0.1.1 复现；不要将其分数与 Terminal-Bench 2 对比。
τ³ 的最终文本在输出之前会先对照确定性的工具证据进行检查，EvalPlus 仍然是 HumanEval+/MBPP+ 的官方评分器。

### 验证快照

以下结果作为工程验证记录保留，而不是单一的榜单数字：各个测试套件衡量的是不同能力，且模型、预算和 verifier 始终是每项结论的一部分。以下分数来自 2026-07-22 完成的修复评估周期。

| 范围 | 结果 | 它证明了什么 |
| --- | --- | --- |
| 确定性核心与 eval 适配器回归 | **73 passed**，耗时 3.20 s；随后针对性的协议回归 **48 passed** | 安全序列化、进度感知的循环检测、脱敏 trace/usage、外部工作区回执、Harbor/ATIF 映射以及 evaluator 适配器在无需真实模型的情况下保持覆盖。 |
| Terminal-Bench 2 / Harbor | 官方 `sanitize-git-repo` harness 在多次重复试验中完成，带有 trace 和 token 归属；reward 为 **0.0** | 适配器、Docker 生命周期、官方 verifier 和产物端到端可用。一次运行暴露并随后验证了上下文裁剪可能孤立工具响应的问题的修复；其余的失败是模型任务策略的失败（探索不足、遗漏文件或非精确替换），而非 harness 失败。 |
| 历史上的 Terminal-Bench 0.1.1 | `tmux-advanced-workflow` 通过了其官方 verifier | 分页器处理、多行 tmux 提交、循环处理和用量聚合在旧版适配器上可用。`fix-git` 仍未能匹配模型生成的逐字节精确的 Markdown 内容；不计为适配器成功。 |
| τ³ retail | **9/10** 任务通过 | 多回合、混合工具的工作流在保守的有状态工具策略下执行。剩余的一次失败是模型最终响应中一个缺乏支持的 SKU 数量声明。 |
| EvalPlus HumanEval+ | base **160/164**（97.56%）；plus **150/164**（91.46%） | 官方完整集代码生成分数。 |
| EvalPlus MBPP+ | base **369/378**（97.62%）；plus **311/378**（82.28%） | 官方完整集代码生成分数。 |
| 固定的 20 任务 complex MBPP 切片 | direct **12/20**；tool-assisted **11/20**；verified repair **12/20** | 当有确定性 verifier 可用时，有界 repair 工作流能恢复到持平水平，但工具使用不足以证明应该取代直接生成作为独立代码的默认策略。 |

在六回合上限下进行的 Terminal-Bench 2 受控重试，将同一任务的轨迹降至
7,169 输入 token 和 574 输出 token（而最初不设上限的运行是 110,089 和
1,824），并在达到上限前到达了目标 secret 文件。随后正常预算的试验确认了
稳定的执行和完整的 Harbor 产物，但未能通过任务：一次 24 回合的运行使用了
精确要求的占位符，但排除了一个已发现的 JSON 文件；一次 32 回合的运行仍然
只搜索而不编辑。这些是改进终端任务策略的有用证据，而非对 benchmark 质量的
论断。公平的策略比较需要相同的正常回合预算和多次试验。

### 从 Dict / YAML / JSON 创建

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

### 环境变量驱动的 Settings

Nonoka 还集成了 `pydantic-settings` 用于框架级配置：

```python
from nonoka.core.config import settings

print(settings.default_model)   # from NONOKA_DEFAULT_MODEL env var
print(settings.openai_api_key)  # from NONOKA_OPENAI_API_KEY env var
```

## 环境要求

- Python >= 3.10

## 许可证

MIT
