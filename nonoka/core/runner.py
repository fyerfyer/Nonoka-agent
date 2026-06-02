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

  Quick-start (all defaults)::

    runner = Runner()
    result = await runner.run(agent, "Hello", deps=None)

  Production usage::

    runner = Runner(
      model="gpt-4o",
      checkpoint="redis",
      memory="in_memory",
    )
  """

  def __init__(
    self,
    model: str | None = None,
    checkpoint: str | CheckpointStore | None = "memory",
    memory: str | MemoryBackend | None = None,
  ):
    # 1. LLM — auto-create the default LiteLLM provider
    self.llm = self._create_llm(model or settings.default_model)

    # 2. Checkpoint store
    self.checkpoint_store = self._resolve_checkpoint(checkpoint)

    # 3. Memory backend
    self.memory_backend = self._resolve_memory(memory)

  # ------------------------------------------------------------------ #
  # Internal helpers for resolving string shorthands or user objects
  # ------------------------------------------------------------------ #

  def _create_llm(self, model: str) -> LiteLLMProvider:
    """Create the default LLM provider (LiteLLM)."""
    return LiteLLMProvider(model=model)

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

  def _create_session(
    self,
    agent: Agent[DepsT, ResultT],
    deps: DepsT,
    session_id: str | None = None,
  ) -> Session:
    sid = session_id or str(uuid.uuid4())
    memory = None
    if self.memory_backend is not None:
      from nonoka.core.memory import WorkingMemory
      memory = WorkingMemory(
        session_id=sid,
        memory_backend=self.memory_backend,
      )
    return Session(session_id=sid, agent=agent, deps=deps, memory=memory)

  # ------------------------------------------------------------------ #
  # Plan generation
  # ------------------------------------------------------------------ #

  async def _generate_plan(self, session: Session, prompt: str) -> Plan:
    # TODO: Call LLM to generate a Plan based on prompt and tools.
    # For now, return an empty plan as a stub.
    return Plan(steps=(), objective=prompt)

  # ------------------------------------------------------------------ #
  # Scheduler selection
  # ------------------------------------------------------------------ #

  def _select_scheduler(self, plan: Plan | None):
    # Late import to prevent circular dependencies
    from nonoka.core.scheduler import ConversationalScheduler, DAGScheduler

    if plan is None or not plan.steps:
      return ConversationalScheduler()
    return DAGScheduler()

  # ------------------------------------------------------------------ #
  # Public execution API
  # ------------------------------------------------------------------ #

  async def run(
    self,
    agent: Agent[DepsT, ResultT],
    prompt: str,
    deps: DepsT,
    session_id: str | None = None,
  ) -> RunResult[ResultT]:
    """Default run: generate a plan and auto-select a scheduler."""
    session = self._create_session(agent, deps, session_id)

    plan = await self._generate_plan(session, prompt)
    session.current_plan = plan

    scheduler = self._select_scheduler(plan)
    return await scheduler.run(session, self)

  async def run_chat(
    self,
    agent: Agent[DepsT, ResultT],
    prompt: str,
    deps: DepsT,
  ) -> RunResult[ResultT]:
    """Force conversational mode (ReAct)."""
    session = self._create_session(agent, deps)
    from nonoka.core.scheduler import ConversationalScheduler
    scheduler = ConversationalScheduler()
    return await scheduler.run(session, self, prompt=prompt)

  async def run_plan(
    self,
    agent: Agent[DepsT, ResultT],
    plan: Plan,
    deps: DepsT,
  ) -> RunResult[ResultT]:
    """Force execution of a user-defined Plan via DAGScheduler."""
    session = self._create_session(agent, deps)
    session.current_plan = plan
    from nonoka.core.scheduler import DAGScheduler
    scheduler = DAGScheduler()
    return await scheduler.run_plan(session, self)

  async def resume(
    self,
    agent: Agent[DepsT, ResultT],
    session_id: str,
    deps: DepsT,
  ) -> RunResult[ResultT]:
    """Resume execution from a checkpoint."""
    state = await self.checkpoint_store.load_session(session_id)
    if not state:
      return RunResult(success=False, error=f"Session {session_id} not found in checkpoint store.")

    session = Session.from_state(state, agent, deps=deps)

    if session.status in {SessionStatus.COMPLETED, SessionStatus.FAILED}:
      return RunResult(success=session.status == SessionStatus.COMPLETED, session=session)

    scheduler = self._select_scheduler(session.current_plan)

    if hasattr(scheduler, "resume"):
      return await scheduler.resume(session, self)
    return await scheduler.run(session, self)
