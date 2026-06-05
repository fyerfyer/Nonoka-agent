import uuid
from typing import Any, TypeVar, Generic

from nonoka.core.agent import Agent
from nonoka.core.session import Session, SessionStatus
from nonoka.core.plan import Plan
from nonoka.core.types import RunResult
from nonoka.core.checkpoint import CheckpointStore
from nonoka.core.memory import MemoryBackend
from nonoka.core.config import settings
from nonoka.core.llm import LiteLLMProvider

DepsT = TypeVar("DepsT")
ResultT = TypeVar("ResultT")


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

  Quick-start (all defaults)::

    runner = Runner()
    result = await runner.run_react(agent, "Hello", deps=None)

  Production usage::

    runner = Runner(
      checkpoint="redis",
      memory="in_memory",
    )
  """

  def __init__(
    self,
    checkpoint: str | CheckpointStore | None = "memory",
    memory: str | MemoryBackend | None = None,
  ):
    # LLM providers are cached per-model and created lazily on first use.
    self._llm_cache: dict[str, LiteLLMProvider] = {}

    # 2. Checkpoint store
    self.checkpoint_store = self._resolve_checkpoint(checkpoint)

    # 3. Memory backend
    self.memory_backend = self._resolve_memory(memory)

  # ------------------------------------------------------------------ #
  # LLM provider cache — created on demand per agent.model
  # ------------------------------------------------------------------ #

  # Current active LLM provider (set by _ensure_llm for backward compatibility)
  llm: LiteLLMProvider | None = None  # type: ignore[misc]

  def _ensure_llm(self, agent: Agent[DepsT, ResultT]) -> LiteLLMProvider:
    """Return a cached LLM provider for *agent.model*, creating one if needed."""
    model = agent.model
    if model in self._llm_cache:
      self.llm = self._llm_cache[model]
      return self.llm

    provider = self._create_llm(model)
    self._llm_cache[model] = provider
    self.llm = provider
    return provider

  def _create_llm(self, model: str) -> LiteLLMProvider:
    """Create the default LLM provider (LiteLLM)."""
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
    )

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
    if checkpoint is None or checkpoint == "memory":
      from nonoka.backends.checkpoint.memory import MemoryCheckpointStore
      return MemoryCheckpointStore()
    if checkpoint == "redis":
      from nonoka.backends.checkpoint.redis import RedisCheckpointStore
      return RedisCheckpointStore()
    # Duck-typing: accept any object that quacks like a CheckpointStore
    self._validate_callable(
      checkpoint, "CheckpointStore",
      ["save_session", "load_session", "save_step_status", "save_step_result", "save_step_error"]
    )
    return checkpoint  # type: ignore[return-value]

  def _resolve_memory(self, memory: str | MemoryBackend | None) -> MemoryBackend | None:
    if memory is None:
      return None
    if memory == "in_memory":
      from nonoka.backends.memory.in_memory import InMemoryBackend
      return InMemoryBackend()
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
    memory = None
    if self.memory_backend is not None:
      from nonoka.core.memory import WorkingMemory
      memory = WorkingMemory(
        session_id=sid,
        memory_backend=self.memory_backend,
      )

    session = Session(session_id=sid, agent=agent, deps=deps, memory=memory)

    # Inherit memory from parent session if requested
    if parent_session_id is not None:
      await self._inherit_memory(parent_session_id, session)

    return session

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
    return await paradigm.run(session, self, prompt=prompt)

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
    return await executor.execute(plan, session, self)

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
    return await reflective.run(session, self, prompt=prompt)

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
