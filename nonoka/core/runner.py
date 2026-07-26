from __future__ import annotations

import asyncio
import uuid
import weakref
from collections.abc import AsyncIterator
from typing import Any, Callable, TypeVar

from pydantic import BaseModel, Field

from nonoka.core.agent import Agent
from nonoka.core.session import Session, SessionStatus
from nonoka.core.plan import Plan
from nonoka.core.types import RunResult
from nonoka.core.checkpoint import CheckpointStore
from nonoka.core.memory import MemoryBackend
from nonoka.core.config import settings
from nonoka.core.llm import LiteLLMProvider, CircuitBreaker
from nonoka.core.hooks import Hooks, HookContext

DepsT = TypeVar("DepsT")
ResultT = TypeVar("ResultT")


class _Unset:
  """Sentinel to distinguish "user passed None" from "user passed nothing"."""


_UNSET = _Unset()


class Runner:
  """Stateless execution coordinator.

  Runner owns all runtime components (LLM, Checkpoint, Memory) and is
  responsible for assembling them.  Agent remains a pure configuration
  object.

  The user **explicitly** chooses the execution paradigm via one of the
  ``run_*`` methods.  There is no automatic scheduler selection.

  **Model resolution** — the LLM model is taken from ``agent.model``, not
  from ``Runner`` construction.  This eliminates the ambiguity of having
  two places to specify the model.

  **Resilience** — retry, timeout, and circuit-breaker configuration is
  pulled from ``agent.default_retry`` and ``agent.default_timeout`` so
  that each Agent can declare its own reliability policy.

  Quick-start (all defaults)::

    runner = Runner()
    result = await runner.run_react(agent, "Hello", deps=None)

  Memory-only (testing)::

    runner = Runner(checkpoint="memory")

  Streaming usage (CLI)::

    async for event in runner.run_react_stream(agent, "Hello", deps=None):
        if event.type == "content_delta":
            print(event.data["content"], end="", flush=True)
  """

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> "Runner":
    """Construct a ``Runner`` from a plain dictionary.

    Example::

      runner = Runner.from_dict({
        "checkpoint": "redis",
        "memory": "in_memory",
      })
    """
    return cls(**data)

  def __init__(
    self,
    checkpoint: str | CheckpointStore | None = None,
    memory: str | MemoryBackend | None | _Unset = _UNSET,
    circuit_breaker: CircuitBreaker | None = None,
    hooks: Hooks | None = None,
    gateway: Any | None = None,
    event_store: Any | None = None,
    observability: Any | None = None,
    response_cache: Any | None = None,
    cache_ttl_seconds: int = 604800,
    semantic_cache: Any | None = None,
    semantic_embedder: Any | None = None,
    semantic_threshold: float = 0.92,
    cache_namespace: str | Callable[[], str | None] = "default",
  ):
    # LLM providers are cached per-model and created lazily on first use.
    self._llm_cache: dict[str, LiteLLMProvider] = {}

    # Optional circuit breaker shared across all providers created by this runner.
    self._circuit_breaker = circuit_breaker
    self.response_cache = response_cache
    self.cache_ttl_seconds = cache_ttl_seconds
    self.semantic_cache = semantic_cache
    self.semantic_embedder = semantic_embedder
    self.semantic_threshold = semantic_threshold
    self.cache_namespace = cache_namespace
    self._cache_flights: dict[str, asyncio.Future[Any]] = {}

    # 2. Checkpoint store (default: SQLite persistent)
    self.checkpoint_store = self._resolve_checkpoint(checkpoint)

    # 3. Memory backend (default: SQLite persistent, None = disabled)
    self.memory_backend = self._resolve_memory(memory)

    # 4. Hooks / middleware
    self.hooks = hooks or Hooks()
    if observability is not None and event_store is not None:
      raise ValueError("Pass observability or event_store, not both")
    self.observability = observability or event_store
    self.event_store = event_store  # Backward-compatible attribute.
    if self.observability is not None:
      # Add telemetry listeners without replacing user-provided hooks.
      from nonoka.observability import ObservabilityHooks
      observers = ObservabilityHooks(self.observability)
      for point, listeners in observers._store.items():
        self.hooks._store.setdefault(point, []).extend(listeners)

    # 5. Optional Gateway(s) for reverse-channel (Agent-initiated push)
    self._gateways: list[Any] = []
    if gateway is not None:
      self._gateways.append(gateway)

  # ------------------------------------------------------------------ #
  # LLM provider cache — created on demand per agent.model
  # ------------------------------------------------------------------ #

  # Current active LLM provider (set by _ensure_llm for backward compatibility)
  llm: LiteLLMProvider | None = None  # type: ignore[misc]

  # ------------------------------------------------------------------ #
  # Gateway management — supports multiple gateways without overwriting
  # ------------------------------------------------------------------ #

  @property
  def gateway(self) -> Any | None:
    """Return the most recently added Gateway bound to this runner, or None."""
    return self._gateways[-1] if self._gateways else None

  @gateway.setter
  def gateway(self, value: Any | None) -> None:
    """Replace all gateways with a single one (backward-compatible setter)."""
    self._gateways.clear()
    if value is not None:
      self._gateways.append(value)

  def add_gateway(self, gateway: Any) -> None:
    """Add a Gateway to this runner without overwriting existing ones."""
    if gateway not in self._gateways:
      self._gateways.append(gateway)

  def _ensure_llm(self, agent: Agent[DepsT, ResultT]) -> LiteLLMProvider:
    """Return a cached LLM provider for *agent.model*, creating one if needed."""
    model = agent.model
    if model in self._llm_cache:
      self.llm = self._llm_cache[model]
      return self.llm

    provider = self._create_llm(agent)
    self._llm_cache[model] = provider
    self.llm = provider
    return provider

  def _create_llm(self, agent: Agent[DepsT, ResultT]) -> LiteLLMProvider:
    """Create the default LLM provider (LiteLLM) bound to *agent*'s policy."""
    model = agent.model
    # Pass API key / base_url from settings so .env overrides work
    api_key = settings.openai_api_key
    base_url = settings.openai_base_url
    # Also support generic api_key / base_url from env without prefix
    import os
    if not api_key:
      api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if not base_url:
      base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL")

    # LiteLLM needs a provider prefix when a custom base_url is used.
    # Deepseek via OpenAI-compatible endpoint → openai/deepseek-chat
    # Only add prefix if the model string does not already contain one.
    if base_url and "/" not in model:
      model = f"openai/{model}"

    return LiteLLMProvider(
      model=model,
      api_key=api_key,
      base_url=base_url,
      retry_policy=agent.default_retry,
      timeout=agent.default_timeout,
      circuit_breaker=self._circuit_breaker,
    )

  async def record_llm_usage(
    self,
    session: Session,
    usage: dict[str, Any] | None,
    *,
    cache_hit: bool = False,
  ) -> None:
    """Record usage once, then expose it through a first-class hook.

    Providers are intentionally not allowed to mutate runtime usage directly:
    this keeps accounting identical for streamed, non-streamed and future
    cached completions.
    """
    normalized = dict(usage or {})
    normalized["cache_hit"] = cache_hit
    session.record_model_usage(normalized, cache_hit=cache_hit)
    await self.hooks.emit_llm_usage(HookContext(session=session, runner=self), normalized)

  async def complete(
    self, *, messages: list[Any], tools: list[dict[str, Any]] | None,
    temperature: float | None, max_tokens: int | None,
  ) -> Any:
    """Run a cache-safe completion through a single-flight exact cache."""
    provider = self.llm
    if provider is None:
      raise RuntimeError("Runner LLM has not been initialized")
    eligible = self.response_cache is not None and not tools and not any(
      str(getattr(message, "role", "")) == "LLMMessageRole.TOOL" or str(getattr(message, "role", "")) == "tool"
      for message in messages
    )
    if not eligible:
      return await provider.chat(messages=messages, tools=tools, temperature=temperature, max_tokens=max_tokens)
    namespace = self.cache_namespace() if callable(self.cache_namespace) else self.cache_namespace
    # A scope resolver may deliberately return None when its workspace no
    # longer has a trustworthy revision fingerprint.  Exact matching remains
    # safe; semantic reuse is disabled for that turn.
    namespace = namespace or "default"
    from nonoka.core.cache import canonical_response_key
    key = canonical_response_key(model=provider.model, messages=messages, tools=tools,
      temperature=temperature, max_tokens=max_tokens, namespace=namespace)
    hit = await self.response_cache.get(key)
    if hit is not None:
      hit.usage["_cache_hit"] = True
      hit.usage["_cache_source"] = "exact"
      return hit
    semantic_vector = None
    if self.semantic_cache is not None and self.semantic_embedder is not None and namespace != "default" and temperature in (None, 0):
      from nonoka.core.cache import semantic_cache_query, semantic_cache_variant
      query = semantic_cache_query(messages)
      if query is None:
        # A multi-turn exchange has hidden conversational state. Exact cache
        # remains available, but semantic reuse would be unsound here.
        query = None
      try:
        if query is not None:
          semantic_vector = await self.semantic_embedder.embed(query)
          semantic_hit = await self.semantic_cache.get(
            semantic_vector, model=provider.model, scope=namespace,
            variant=semantic_cache_variant(model=provider.model, temperature=temperature, max_tokens=max_tokens),
            threshold=self.semantic_threshold,
          )
          if semantic_hit is not None:
            semantic_hit.usage["_cache_hit"] = True
            semantic_hit.usage["_semantic_cache_hit"] = True
            semantic_hit.usage["_cache_source"] = "semantic"
            return semantic_hit
      except Exception:
        # Semantic caching is an optional cost optimization, never an LLM availability dependency.
        semantic_vector = None
    flight = self._cache_flights.get(key)
    if flight is not None:
      return await flight
    flight = asyncio.get_running_loop().create_future()
    self._cache_flights[key] = flight
    try:
      response = await provider.chat(messages=messages, tools=tools, temperature=temperature, max_tokens=max_tokens)
      if not response.tool_calls:
        await self.response_cache.put(key, response, self.cache_ttl_seconds)
        if semantic_vector is not None:
          try:
            await self.semantic_cache.put(
              semantic_vector, response, model=provider.model, scope=namespace,
              variant=semantic_cache_variant(model=provider.model, temperature=temperature, max_tokens=max_tokens),
              ttl_seconds=self.cache_ttl_seconds,
            )
          except Exception:
            # A cache write must never turn a successful provider response into an error.
            pass
      flight.set_result(response)
      return response
    except BaseException as exc:
      flight.set_exception(exc)
      raise
    finally:
      self._cache_flights.pop(key, None)

  # ------------------------------------------------------------------ #
  # Backend resolution helpers
  # ------------------------------------------------------------------ #

  @staticmethod
  def _validate_callable(obj: Any, name: str, methods: list[str]) -> None:
    """Duck-typing sanity check: verify *obj* has the required callable *methods*."""
    missing = [
      m for m in methods
      if not callable(getattr(obj, m, None))
    ]
    if missing:
      raise TypeError(
        f"{name} is missing required methods: {', '.join(missing)}"
      )

  def _resolve_checkpoint(self, checkpoint: str | CheckpointStore | None) -> CheckpointStore:
    if checkpoint is None:
      # Default: SQLite persistent store
      from nonoka.backends.checkpoint.sqlite import SQLiteCheckpointStore
      return SQLiteCheckpointStore()
    if checkpoint == "memory":
      from nonoka.backends.checkpoint.memory import MemoryCheckpointStore
      return MemoryCheckpointStore()
    if checkpoint == "disabled":
      from nonoka.backends.checkpoint.noop import NoOpCheckpointStore
      return NoOpCheckpointStore()
    # Duck-typing: accept any object that quacks like a CheckpointStore
    self._validate_callable(
      checkpoint, "CheckpointStore",
      ["save_session", "load_session", "save_step_status", "save_step_result", "save_step_error"]
    )
    return checkpoint  # type: ignore[return-value]

  def _resolve_memory(self, memory: str | MemoryBackend | None | _Unset) -> MemoryBackend | None:
    if isinstance(memory, _Unset):
      # Default: SQLite persistent backend
      from nonoka.backends.memory.sqlite import SQLiteMemoryBackend
      return SQLiteMemoryBackend()
    if memory is None or memory == "disabled":
      return None
    if memory == "in_memory":
      from nonoka.backends.memory.in_memory import InMemoryBackend
      return InMemoryBackend()
    # Duck-typing: accept any object that quacks like a MemoryBackend
    self._validate_callable(
      memory, "MemoryBackend",
      ["add", "search", "get_history", "get_user_memory"]
    )
    return memory  # type: ignore[return-value]

  # ------------------------------------------------------------------ #
  # Session lifecycle
  # ------------------------------------------------------------------ #

  async def _create_session(
    self,
    agent: Agent[DepsT, ResultT],
    deps: DepsT,
    session_id: str | None = None,
    parent_session_id: str | None = None,
  ) -> Session:
    sid = session_id or str(uuid.uuid4())
    from nonoka.core.memory import WorkingMemory
    state = await self.checkpoint_store.load_session(session_id) if session_id is not None else None
    persisted_limits = getattr(getattr(state, "runtime_state", None), "limits", None)
    configured_limits = persisted_limits or getattr(agent, "runtime_limits", None)
    memory = WorkingMemory(
      session_id=sid,
      memory_backend=self.memory_backend,
      max_tokens=(getattr(configured_limits, "max_context_tokens", None) or 8192),
    )

    session = (
      Session.from_state(state, agent, deps=deps, memory=memory)
      if state is not None
      else Session(session_id=sid, agent=agent, deps=deps, memory=memory)
    )

    # Bind runner so AgentTool can access it via ctx.session._runner_ref
    object.__setattr__(session, "_runner_ref", weakref.ref(self))

    # Bind gateway for reverse-channel push (tools can access ctx.gateway)
    if self._gateways:
      # Bind the primary (first) gateway for reverse-channel push
      object.__setattr__(session, "_gateway_ref", weakref.ref(self._gateways[0]))

    # Inherit memory from parent session if requested
    if parent_session_id is not None:
      await self._inherit_memory(parent_session_id, session)

    return session

  async def cancel_session(self, session_id: str) -> bool:
    """Persist cancellation so a later process cannot resume the session."""
    state = await self.checkpoint_store.load_session(session_id)
    if state is None:
      return False
    session = Session.from_state(state, agent=Agent(model="cancelled"), memory=None)
    session.cancel()
    session.status = SessionStatus.CANCELLED
    session.end_time = __import__("datetime").datetime.now()
    await self.checkpoint_store.save_session(session_id, session.to_state())
    return True

  @staticmethod
  def _attach_trace(result: RunResult) -> RunResult:
    """Expose the session trace without making callers reach into Session."""
    if result.session is not None and hasattr(result.session, "trace"):
      result.session.trace.finish(
        success=result.success, error_type=result.error_type, error=result.error,
      )
      result.trace = result.session.trace.to_dict()
    return result

  async def _inherit_memory(self, parent_session_id: str, session: Session) -> None:
    """Copy memory entries from a parent session into the child session."""
    parent_state = await self.checkpoint_store.load_session(parent_session_id)
    if not parent_state:
      return

    # Handle both dict and SessionState objects
    if hasattr(parent_state, "memory_entries"):
      memory_entries = parent_state.memory_entries
    elif isinstance(parent_state, dict):
      memory_entries = parent_state.get("memory_entries", [])
    else:
      return

    if not memory_entries:
      return

    from nonoka.core.memory import MemoryEntry, MemoryRole
    for entry_data in memory_entries:
      try:
        entry = MemoryEntry(**entry_data)
        session.memory.entries.append(entry)  # type: ignore[union-attr]
      except Exception:
        # Skip malformed entries
        continue

  # ------------------------------------------------------------------ #
  # Public execution API — explicit paradigm selection
  # ------------------------------------------------------------------ #

  async def run_react(
    self,
    agent: Agent[DepsT, ResultT],
    prompt: str,
    deps: DepsT,
    session_id: str | None = None,
    parent_session_id: str | None = None,
  ) -> RunResult[ResultT]:
    """Execute in **ReAct** (exploratory) mode.

    The LLM re-decides the next action every turn.  Suitable for
    information retrieval, multi-step reasoning, and dynamic branching.
    """
    from nonoka.core.paradigm import ReActAgent
    session = await self._create_session(agent, deps, session_id, parent_session_id)
    # Ensure LLM is ready for this agent's model
    self._ensure_llm(agent)
    paradigm = ReActAgent()
    hook_ctx = HookContext(session=session, runner=self)
    await self.hooks.emit_session_start(hook_ctx)
    result = await paradigm.run(session, self, prompt=prompt)
    from nonoka.core.extensions import LoopExtensionContext, LoopExtensionManager
    await LoopExtensionManager(list(getattr(agent, "extensions", []))).after_run(
      LoopExtensionContext(session=session, runner=self, prompt=prompt, turn=session.turn_count), result,
    )
    result = self._attach_trace(result)
    await self.hooks.emit_session_end(hook_ctx, result)
    return result

  async def run_react_stream(
    self,
    agent: Agent[DepsT, ResultT],
    prompt: str,
    deps: DepsT,
    session_id: str | None = None,
    parent_session_id: str | None = None,
  ) -> AsyncIterator["StreamEvent"]:
    """Execute in **ReAct** mode and yield streaming events.

    This is the CLI-friendly entry point: callers receive content deltas,
    tool-call lifecycle events, and the final result as discrete events
    rather than a single batched ``RunResult``.
    """
    from nonoka.core.paradigm import ReActAgent
    session = await self._create_session(agent, deps, session_id, parent_session_id)
    self._ensure_llm(agent)
    paradigm = ReActAgent()
    hook_ctx = HookContext(session=session, runner=self)
    await self.hooks.emit_session_start(hook_ctx)
    result_data: Any = None
    result_success = False
    result_error: str | None = None
    result_error_type: str | None = None
    try:
      async for event in paradigm.run_stream(session, self, prompt=prompt):
        if event.type == "final":
          result_data = event.data.get("data")
          result_success = event.data.get("success", False)
        elif event.type == "error":
          result_success = False
          result_error = event.data.get("error")
          result_error_type = event.data.get("error_type")
        yield event
    finally:
      result = RunResult(
        success=result_success,
        data=result_data,
        session=session,
        error=result_error,
        error_type=result_error_type,
      )
      await self.hooks.emit_session_end(hook_ctx, result)

  async def resume_approval(
    self,
    agent: Agent[DepsT, ResultT],
    deps: DepsT,
    session_id: str,
    approvals: dict[str, dict[str, Any]],
  ) -> AsyncIterator["StreamEvent"]:
    """Resume a paused ReAct session after a human tool-call approval.

    The session must exist in the checkpoint store and be in the ``PAUSED``
    state.  Approved tools are executed, rejected ones are recorded as errors,
    and the ReAct loop continues from there.
    """
    from nonoka.core.paradigm import ReActAgent
    session = await self._create_session(agent, deps, session_id=session_id)
    self._ensure_llm(agent)
    paradigm = ReActAgent()
    hook_ctx = HookContext(session=session, runner=self)
    await self.hooks.emit_session_start(hook_ctx)
    result_data: Any = None
    result_success = False
    result_error: str | None = None
    result_error_type: str | None = None
    try:
      async for event in paradigm.resume_approval(session, self, approvals):
        if event.type == "final":
          result_data = event.data.get("data")
          result_success = event.data.get("success", False)
        elif event.type == "error":
          result_success = False
          result_error = event.data.get("error")
          result_error_type = event.data.get("error_type")
        yield event
    finally:
      result = RunResult(
        success=result_success,
        data=result_data,
        session=session,
        error=result_error,
        error_type=result_error_type,
      )
      await self.hooks.emit_session_end(hook_ctx, result)

  async def resume_external_tools(
    self,
    agent: Agent[DepsT, ResultT],
    deps: DepsT,
    session_id: str,
    results: dict[str, Any],
  ) -> AsyncIterator["StreamEvent"]:
    """Resume a paused ReAct session after external tool execution.

    The session must exist in the checkpoint store and be in the ``PAUSED``
    state. Tool results supplied by the external host are injected into memory,
    and the ReAct loop continues from there.
    """
    from nonoka.core.paradigm import ReActAgent
    session = await self._create_session(agent, deps, session_id=session_id)
    self._ensure_llm(agent)
    paradigm = ReActAgent()
    hook_ctx = HookContext(session=session, runner=self)
    await self.hooks.emit_session_start(hook_ctx)
    result_data: Any = None
    result_success = False
    result_error: str | None = None
    result_error_type: str | None = None
    try:
      async for event in paradigm.resume_external_tools(session, self, results):
        if event.type == "final":
          result_data = event.data.get("data")
          result_success = event.data.get("success", False)
        elif event.type == "error":
          result_success = False
          result_error = event.data.get("error")
          result_error_type = event.data.get("error_type")
        yield event
    finally:
      result = RunResult(
        success=result_success,
        data=result_data,
        session=session,
        error=result_error,
        error_type=result_error_type,
      )
      await self.hooks.emit_session_end(hook_ctx, result)

  async def run_plan(
    self,
    agent: Agent[DepsT, ResultT],
    plan: Plan,
    deps: DepsT,
    session_id: str | None = None,
    parent_session_id: str | None = None,
  ) -> RunResult[ResultT]:
    """Execute a user-defined **Plan** via PlanExecutor (deterministic).

    No LLM calls — just topological sort + parallel layer execution.
    Suitable for CI/CD pipelines, ETL, and other known workflows.
    """
    from nonoka.core.paradigm import PlanExecutor
    session = await self._create_session(agent, deps, session_id, parent_session_id)
    self._ensure_llm(agent)
    executor = PlanExecutor()
    hook_ctx = HookContext(session=session, runner=self)
    await self.hooks.emit_session_start(hook_ctx)
    result = self._attach_trace(await executor.execute(plan, session, self))
    await self.hooks.emit_session_end(hook_ctx, result)
    return result

  async def run_reflective(
    self,
    agent: Agent[DepsT, ResultT],
    evaluator: Any,
    prompt: str,
    deps: DepsT,
    max_iterations: int = 3,
    session_id: str | None = None,
    parent_session_id: str | None = None,
  ) -> RunResult[ResultT]:
    """Execute in **Reflective** (quality-driven) mode.

    Actor → Evaluate → Revise loop.  The *evaluator* decides whether the
    result is good enough; if not, feedback is injected for another try.

    Args:
      agent: The Agent configuration (tools, system prompt, etc.).
      evaluator: An object implementing the ``Evaluator`` protocol (or a
        ``ToolEvaluator`` wrapper around a validation tool).
      prompt: The task description.
      deps: Dependency object injected into tools via ``RunContext``.
      max_iterations: Maximum Actor → Evaluate cycles.
      session_id: Optional existing session ID for resuming.
    """
    from nonoka.core.paradigm import ReActAgent, ReflectiveAgent
    session = await self._create_session(agent, deps, session_id, parent_session_id)
    self._ensure_llm(agent)
    actor = ReActAgent()
    reflective = ReflectiveAgent(
      actor=actor,
      evaluator=evaluator,
      max_iterations=max_iterations,
    )
    hook_ctx = HookContext(session=session, runner=self)
    await self.hooks.emit_session_start(hook_ctx)
    result = await reflective.run(session, self, prompt=prompt)
    from nonoka.core.extensions import LoopExtensionContext, LoopExtensionManager
    await LoopExtensionManager(list(getattr(agent, "extensions", []))).after_run(
      LoopExtensionContext(session=session, runner=self, prompt=prompt, turn=session.turn_count), result,
    )
    result = self._attach_trace(result)
    await self.hooks.emit_session_end(hook_ctx, result)
    return result

  # ------------------------------------------------------------------ #
  # Legacy alias — defaults to ReAct
  # ------------------------------------------------------------------ #

  async def run(
    self,
    agent: Agent[DepsT, ResultT],
    prompt: str,
    deps: DepsT,
    session_id: str | None = None,
    parent_session_id: str | None = None,
  ) -> RunResult[ResultT]:
    """Default entry-point — runs in **ReAct** mode.

    .. deprecated::
      Use ``run_react`` for explicitness.  This alias is kept for
      backward compatibility.
    """
    return await self.run_react(agent, prompt, deps, session_id, parent_session_id)

  # ------------------------------------------------------------------ #
  # Resume
  # ------------------------------------------------------------------ #

  async def resume(
    self,
    agent: Agent[DepsT, ResultT],
    session_id: str,
    deps: DepsT,
  ) -> RunResult[ResultT]:
    """Resume execution from a checkpoint.

    The session's ``current_plan`` determines which paradigm to resume:
    * If a Plan exists → resume via ``PlanExecutor``.
    * Otherwise → resume via ``ReActAgent``.
    """
    state = await self.checkpoint_store.load_session(session_id)
    if not state:
      return RunResult(success=False, error=f"Session {session_id} not found in checkpoint store.")

    # Re-create WorkingMemory so that memory_entries can be restored
    memory = None
    if self.memory_backend is not None:
      from nonoka.core.memory import WorkingMemory
      memory = WorkingMemory(
        session_id=session_id,
        memory_backend=self.memory_backend,
      )

    session = Session.from_state(state, agent, deps=deps, memory=memory)

    if session.status in {SessionStatus.COMPLETED, SessionStatus.FAILED}:
      return RunResult(success=session.status == SessionStatus.COMPLETED, session=session)

    self._ensure_llm(agent)

    # Route to the correct paradigm based on whether a plan was in flight
    if session.current_plan and session.current_plan.steps:
      from nonoka.core.paradigm import PlanExecutor
      executor = PlanExecutor()
      return await executor.resume(session.current_plan, session, self)
    else:
      from nonoka.core.paradigm import ReActAgent
      paradigm = ReActAgent()
      return await paradigm.resume(session, self)


# ------------------------------------------------------------------ #
# Stream event model
# ------------------------------------------------------------------ #

class StreamEvent(BaseModel):
  """Discrete event emitted by ``Runner.run_react_stream()``.

  Types:
  * ``content_delta`` — incremental LLM text.
  * ``tool_call_progress`` — content-free progress while the model streams a
    potentially large tool payload.
  * ``tool_call_start`` — LLM requested one or more tools.
  * ``tool_call_result`` — a tool finished (success or error).
  * ``approval_request`` — a tool call is waiting for human approval.
  * ``final`` — execution finished; ``data`` contains ``RunResult`` fields.
  * ``error`` — a terminal error occurred.
  """
  type: str
  data: dict[str, Any] = Field(default_factory=dict)
