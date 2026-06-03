import uuid
import json
import re
from typing import Any, TypeVar, Generic

from nonoka.core.agent import Agent
from nonoka.core.session import Session, SessionStatus
from nonoka.core.plan import Plan, Step
from nonoka.core.types import RunResult
from nonoka.core.checkpoint import CheckpointStore
from nonoka.core.memory import MemoryBackend
from nonoka.core.config import settings
from nonoka.core.llm import LiteLLMProvider, LLMMessage, LLMMessageRole
from nonoka.core.logger import get_logger

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
    # TODO: other model? 
    if base_url and "/" not in model.split("/")[0]:
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
    """Use the LLM to generate a Plan from the user prompt and available tools.

    The LLM is asked to emit a JSON object describing the execution plan.
    On parse failure we gracefully fall back to an empty plan so the
    runner can still use the ConversationalScheduler.
    """
    logger = get_logger("nonoka.runner")
    tools_info: list[dict[str, Any]] = []
    for tool in session.agent.tools:
      tools_info.append({
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
      })

    # Build the planning prompt with strong schema guidance.
    system_prompt = (
      "You are a task planner. Given a user request and the list of available tools, "
      "produce a structured execution plan as JSON.\n\n"
      "Rules:\n"
      "1. Use ONLY the tools listed below. Do NOT invent tools.\n"
      "2. Each step must have a unique `id` (use snake_case).\n"
      "3. `args` must be valid JSON values.\n"
      "4. If a step needs the output of a previous step, put the referenced step id in `depends_on`.\n"
      "5. If the task is purely conversational or needs no tools, return an empty `steps` array.\n"
      "6. If the task requires only a single tool call, return exactly one step.\n"
      "7. When a step argument needs to reference the output of a previous step, "
      "wrap it in a JSON STRING like this: \"ref(\\\"step_id\\\")\" or "
      "\"ref(\\\"step_id\\\", \\\"path\\\")\". The ref marker must be a quoted string value, "
      "not raw code. Example: {\"result\": \"ref(\\\"calc\\\", \\\"result\\\")\"}.\n\n"
      "JSON schema:\n"
      '{\n'
      '  "objective": "brief description of the overall goal",\n'
      '  "steps": [\n'
      '    {\n'
      '      "id": "step_id",\n'
      '      "tool": "tool_name",\n'
      '      "args": {"arg_name": "value"},\n'
      '      "depends_on": ["previous_step_id"]\n'
      '    }\n'
      '  ]\n'
      '}\n\n'
    )
    if tools_info:
      system_prompt += f"Available tools:\n{json.dumps(tools_info, indent=2, ensure_ascii=False)}\n"
    else:
      system_prompt += "No tools available.\n"

    messages = [
      LLMMessage(role=LLMMessageRole.SYSTEM, content=system_prompt),
      LLMMessage(role=LLMMessageRole.USER, content=prompt),
    ]

    try:
      response = await self.llm.chat(
        messages=messages,
        temperature=0.1,
        max_tokens=2000,
      )
      content = response.content or "{}"

      # Try to extract JSON from a markdown code block first.
      m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
      if m:
        content = m.group(1)

      plan_data = json.loads(content)
    except Exception as exc:
      logger.warning(
        "plan_generation_parse_failed",
        session_id=session.session_id,
        error=str(exc),
        raw_content=getattr(response, "content", None),
      )
      return Plan(steps=(), objective=prompt)

    # --- Parse steps ---
    from nonoka.core.plan import ref as plan_ref
    raw_steps = plan_data.get("steps", [])
    steps: list[Step] = []
    for s in raw_steps:
      try:
        step_id = s["id"]
        tool_name = s["tool"]
        args = s.get("args", {})
        depends_on = frozenset(s.get("depends_on", []))

        # Validate: tool must exist in agent's toolkit
        known = [t.name for t in session.agent.tools]
        if tool_name not in known:
          logger.warning(
            "plan_generated_unknown_tool",
            session_id=session.session_id,
            step_id=step_id,
            tool_name=tool_name,
            known_tools=known,
          )
          # We still keep the step; the scheduler will report the error at runtime.

        # Convert any string ref markers into Ref objects
        parsed_args: dict[str, Any] = {}
        for k, v in args.items():
          if isinstance(v, str) and v.startswith("ref("):
            # Accept ref("step_id") or ref("step_id", "path")
            rm = re.match(r'ref\("([^"]+)"(?:\s*,\s*"([^"]+)")?\)', v)
            if rm:
              ref_step = rm.group(1)
              ref_path = rm.group(2) or ""
              parsed_args[k] = plan_ref(ref_step, ref_path) if ref_path else plan_ref(ref_step)
            else:
              parsed_args[k] = v
          else:
            parsed_args[k] = v

        steps.append(Step(
          id=step_id,
          tool=tool_name,
          args=parsed_args,
          depends_on=depends_on,
        ))
      except Exception as exc:
        logger.warning(
          "plan_generation_step_parse_failed",
          session_id=session.session_id,
          step_data=s,
          error=str(exc),
        )
        continue

    objective = plan_data.get("objective", prompt)
    plan = Plan(steps=tuple(steps), objective=objective)

    logger.info(
      "plan_generated",
      session_id=session.session_id,
      num_steps=len(steps),
      objective=objective,
    )
    return plan

  # ------------------------------------------------------------------ #
  # Scheduler selection
  # ------------------------------------------------------------------ #

  def _select_scheduler(self, plan: Plan | None):
    """Hybrid scheduler selection.

    * No plan / empty plan  → ConversationalScheduler (pure chat)
    * Any plan with steps   → DAGScheduler (execute the planned steps)
    """
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
    # Pass prompt through so conversational mode can seed the memory
    if hasattr(scheduler, "run"):
      sig = __import__("inspect").signature(scheduler.run)
      if "prompt" in sig.parameters:
        return await scheduler.run(session, self, prompt=prompt)
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

    scheduler = self._select_scheduler(session.current_plan)

    if hasattr(scheduler, "resume"):
      return await scheduler.resume(session, self)
    return await scheduler.run(session, self)
