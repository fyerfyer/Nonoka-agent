import asyncio
import json
import re
from typing import Any, Protocol, runtime_checkable

from nonoka.core.session import Session, SessionStatus, StepStatus, StepResult, StepFailure
from nonoka.core.types import RunResult
from nonoka.core.context import RunContext
from nonoka.core.plan import Step, Ref
from nonoka.core.event import AgentEvent, EventType
from nonoka.core.logger import get_logger
from nonoka.core.llm import LLMMessage, LLMMessageRole
from nonoka.core.memory import MemoryRole
from nonoka.core.errors import (
  ErrorPolicy,
  SafetyError,
  MaxTurnsExceeded,
  ToolErrorActionType,
)

_logger = get_logger("nonoka.scheduler")


# --------------------------------------------------------------------------- #
# Scheduler Protocol (internal only — not exported to public API)
# --------------------------------------------------------------------------- #

@runtime_checkable
class Scheduler(Protocol):
  async def run(self, session: Session, runner: Any, **kwargs) -> RunResult:
    ...
  async def resume(self, session: Session, runner: Any) -> RunResult:
    ...


# --------------------------------------------------------------------------- #
# Ref resolution helpers
# --------------------------------------------------------------------------- #

def _resolve_path(data: Any, path: str) -> Any:
  """Resolve a dot-separated path (optionally with array indices) from *data*."""
  if not path:
    return data
  current = data
  for part in path.split("."):
    if current is None:
      return None

    # Handle bracket indexing: users[0], items[1][2]
    m = re.match(r"^([^\[]+)((?:\[\d+\])+)$", part)
    if m:
      key = m.group(1)
      indices = [int(x) for x in re.findall(r"\[(\d+)\]", m.group(2))]
      current = current[key] if isinstance(current, dict) else getattr(current, key, None)
      for idx in indices:
        if current is None:
          return None
        current = current[idx]
      continue

    if isinstance(current, dict):
      try:
        current = current[part]
      except KeyError:
        return None
    else:
      current = getattr(current, part, None)
  return current


def _resolve_refs(data: Any, completed_steps: dict[str, StepResult]) -> Any:
  """Replace ``Ref`` markers with actual values from *completed_steps*.

  Recursively walks dicts and lists so refs nested at any depth are
  resolved (e.g. ``{"data": {"sum": ref("calc", "result")}}``).
  """
  if isinstance(data, Ref):
    source = completed_steps.get(data.step_id)
    if source is None:
      raise ValueError(
        f"Step '{data.step_id}' not found in completed_steps "
        f"(needed by ref)"
      )
    return _resolve_path(source.data, data.path)

  if isinstance(data, dict):
    return {k: _resolve_refs(v, completed_steps) for k, v in data.items()}

  if isinstance(data, list):
    return [_resolve_refs(item, completed_steps) for item in data]

  return data


# --------------------------------------------------------------------------- #
# ConversationalScheduler — ReAct loop with intra-turn parallelism
# --------------------------------------------------------------------------- #

class ConversationalScheduler:
  """
  ReAct loop, sequential execution between turns.
  Parallel tool calls inside a single turn.
  """

  def __init__(self, error_policy: ErrorPolicy | None = None):
    self.error_policy = error_policy or ErrorPolicy()

  async def run(
    self,
    session: Session,
    runner: Any,
    prompt: str = "",
  ) -> RunResult:
    session.status = SessionStatus.RUNNING
    await runner.checkpoint_store.save_session(session.session_id, session.to_state())

    # Seed the conversation with the user prompt
    if prompt and session.memory is not None:
      await session.memory.add(prompt, MemoryRole.USER)

    try:
      for turn in range(session.agent.max_turns):
        session.turn_count = turn + 1

        # Build messages from memory
        messages = self._build_messages(session)

        # Convert tools to OpenAI function schema
        tools = [t.to_json_schema() for t in session.agent.tools] if session.agent.tools else None

        # --- LLM call ------------------------------------------------
        response = await runner.llm.chat(
          messages=messages,
          tools=tools or None,
        )

        _logger.info(
          "llm.response",
          session_id=session.session_id,
          turn=turn + 1,
          has_tool_calls=bool(response.tool_calls),
        )

        # --- No tool calls → final answer ---------------------------
        if not response.tool_calls:
          content = response.content or ""

          if session.memory is not None:
            await session.memory.add(content, MemoryRole.ASSISTANT)

          # result_type parsing
          parsed_data: Any = content
          if session.agent.result_type is not None:
            parsed_data = await self._parse_result_type(
              session, runner, content, turn
            )
            if parsed_data is None:
              # Parsing failed and max retries exhausted inside helper
              continue

          session.status = SessionStatus.COMPLETED
          session.end_time = __import__("datetime").datetime.now()
          await runner.checkpoint_store.save_session(session.session_id, session.to_state())
          return RunResult(success=True, data=parsed_data, session=session)

        # --- Tool calls → add assistant msg, execute tools ---------
        assistant_msg = LLMMessage(
          role=LLMMessageRole.ASSISTANT,
          content=response.content,
          tool_calls=response.tool_calls,
        )
        if session.memory is not None:
          await session.memory.add(
            response.content or "",
            MemoryRole.ASSISTANT,
            tool_calls=response.tool_calls,
          )

        # Execute tool calls in parallel
        tool_results = await asyncio.gather(*[
          self._execute_tool_call(session, runner, tc)
          for tc in response.tool_calls
        ], return_exceptions=True)

        # Add observations to memory
        for tc, tr in zip(response.tool_calls, tool_results):
          tc_id = tc.get("id") or tc.get("tool_call_id", "unknown")
          if isinstance(tr, Exception):
            obs_text = f"Error: {type(tr).__name__}: {tr}"
          else:
            obs_text = json.dumps(tr, ensure_ascii=False, default=str) if not isinstance(tr, str) else tr

          if session.memory is not None:
            await session.memory.add(
              obs_text,
              MemoryRole.TOOL,
              tool_call_id=tc_id,
            )

        # Checkpoint after each turn
        await runner.checkpoint_store.save_session(session.session_id, session.to_state())

      # Max turns exceeded
      raise MaxTurnsExceeded(
        f"Max turns ({session.agent.max_turns}) exceeded for session {session.session_id}"
      )

    except MaxTurnsExceeded as e:
      session.status = SessionStatus.FAILED
      await runner.checkpoint_store.save_session(session.session_id, session.to_state())
      return RunResult(success=False, session=session, error=str(e))

    except Exception as e:
      session.status = SessionStatus.FAILED
      await runner.checkpoint_store.save_session(session.session_id, session.to_state())
      return RunResult(success=False, session=session, error=str(e))

  async def resume(self, session: Session, runner: Any) -> RunResult:
    """Resume a conversational session from checkpoint."""
    # Conversational resume simply continues the loop
    return await self.run(session, runner, prompt="")

  # ------------------------------------------------------------------ #
  # Internal helpers
  # ------------------------------------------------------------------ #

  def _build_messages(self, session: Session) -> list[LLMMessage]:
    """Convert WorkingMemory entries to LLM messages."""
    messages: list[LLMMessage] = []

    if session.agent.system_prompt:
      messages.append(LLMMessage(role=LLMMessageRole.SYSTEM, content=session.agent.system_prompt))

    if session.memory is not None:
      for entry in session.memory.entries:
        role = entry.role
        if role == MemoryRole.SYSTEM:
          msg_role = LLMMessageRole.SYSTEM
        elif role == MemoryRole.USER:
          msg_role = LLMMessageRole.USER
        elif role == MemoryRole.ASSISTANT:
          msg_role = LLMMessageRole.ASSISTANT
        elif role == MemoryRole.TOOL:
          msg_role = LLMMessageRole.TOOL
        else:
          msg_role = str(role)

        kwargs: dict[str, Any] = {"role": msg_role, "content": entry.content}
        meta = entry.metadata or {}

        # If this was an assistant message with tool_calls, replay them
        if role == MemoryRole.ASSISTANT and meta.get("tool_calls"):
          kwargs["tool_calls"] = meta["tool_calls"]

        # If this was a tool result, attach tool_call_id
        if role == MemoryRole.TOOL and meta.get("tool_call_id"):
          kwargs["tool_call_id"] = meta["tool_call_id"]
          kwargs["name"] = meta.get("tool_name", "")

        messages.append(LLMMessage(**kwargs))

    return messages

  async def _execute_tool_call(
    self,
    session: Session,
    runner: Any,
    tool_call: dict[str, Any],
  ) -> Any:
    """Execute a single tool call with error-policy handling."""
    func_info = tool_call.get("function", {})
    name = func_info.get("name", "")
    arguments = func_info.get("arguments", "{}")

    if isinstance(arguments, str):
      try:
        arguments = json.loads(arguments)
      except json.JSONDecodeError:
        arguments = {}

    capability = next((t for t in session.agent.tools if t.name == name), None)
    if not capability:
      raise ValueError(f"Tool '{name}' not found in agent.")

    ctx = RunContext(session)
    session.step_count += 1

    # Emit event
    ctx.emit(AgentEvent(
      type=EventType.TOOL_CALLED,
      session_id=session.session_id,
      data={"tool": name, "arguments": arguments},
    ))

    result: Any = None
    try:
      result = await capability.invoke(ctx, arguments)
    except Exception as exc:
      action = self.error_policy.on_tool_error(exc, f"{name}_turn_{session.turn_count}")

      if action.type == ToolErrorActionType.RETRY:
        max_retries = max(1, action.kwargs.get("max_retries", 3))
        last_exc = exc
        for attempt in range(max_retries):
          try:
            result = await capability.invoke(ctx, arguments)
            break
          except Exception as retry_exc:
            last_exc = retry_exc
            if attempt == max_retries - 1:
              raise last_exc
        else:
          # All retries exhausted without break — should not reach here
          # because the last iteration raises, but kept for safety.
          raise last_exc
      elif action.type == ToolErrorActionType.REPORT:
        # Return the error as an observation so the LLM can correct
        return {"error": f"{type(exc).__name__}: {exc}"}
      elif action.type == ToolErrorActionType.HALT:
        raise SafetyError(f"Halted on tool error: {exc}") from exc
      else:
        raise

    ctx.emit(AgentEvent(
      type=EventType.TOOL_COMPLETED,
      session_id=session.session_id,
      data={"tool": name, "result_preview": str(result)[:200]},
    ))
    return result

  async def _parse_result_type(
    self,
    session: Session,
    runner: Any,
    content: str,
    current_turn: int,
  ) -> Any | None:
    """Try to parse *content* into ``agent.result_type``.

    Returns the parsed object on success, or ``None`` when retries are
    exhausted (the error has already been injected into memory so the LLM
    can try again on the next turn).
    """
    from pydantic import ValidationError

    result_type = session.agent.result_type
    assert result_type is not None

    # Try JSON parsing first
    data: Any = content
    try:
      data = json.loads(content)
    except json.JSONDecodeError:
      pass  # Maybe the LLM returned plain text that is valid for the model

    try:
      if isinstance(data, dict):
        return result_type(**data)
      return result_type(data)
    except (ValidationError, TypeError) as e:
      err_msg = f"Result parsing failed: {e}. Please return valid JSON matching the expected schema."
      if session.memory is not None:
        await session.memory.add(err_msg, MemoryRole.SYSTEM)
      return None


# --------------------------------------------------------------------------- #
# DAGScheduler — Topological sort + parallel layer execution
# --------------------------------------------------------------------------- #

class DAGScheduler:
  """
  Topological Sort + Parallel Layer Execution.
  """

  def __init__(self, max_concurrency: int = 10):
    self.max_concurrency = max_concurrency

  async def run_plan(self, session: Session, runner: Any) -> RunResult:
    session.status = SessionStatus.RUNNING
    await runner.checkpoint_store.save_session(session.session_id, session.to_state())
    plan = session.current_plan

    if not plan:
      return RunResult(success=False, session=session, error="No plan provided")

    sem = asyncio.Semaphore(self.max_concurrency)

    async def execute_limited(step: Step):
      async with sem:
        return await self._execute_step(step, session, runner)

    try:
      layers = plan.topological_groups()
      for layer_step_ids in layers:
        # Skip steps already completed (resume scenario)
        pending_ids = [
          sid for sid in layer_step_ids
          if sid not in session.completed_steps
        ]
        if not pending_ids:
          continue

        results = await asyncio.gather(*[
          execute_limited(plan.get_step(sid))
          for sid in pending_ids
          if plan.get_step(sid)
        ], return_exceptions=True)

        # Handle failures
        failures = [r for r in results if isinstance(r, Exception)]
        if failures:
          session.status = SessionStatus.FAILED
          await runner.checkpoint_store.save_session(session.session_id, session.to_state())
          return RunResult(
            success=False,
            session=session,
            error=f"Step(s) failed during execution: {failures[0]}",
          )

        # Checkpoint layer
        await runner.checkpoint_store.save_session(session.session_id, session.to_state())

      session.status = SessionStatus.COMPLETED
      await runner.checkpoint_store.save_session(session.session_id, session.to_state())
      return RunResult(success=True, session=session)
    except Exception as e:
      session.status = SessionStatus.FAILED
      await runner.checkpoint_store.save_session(session.session_id, session.to_state())
      return RunResult(success=False, session=session, error=str(e))

  async def _execute_step(self, step: Step, session: Session, runner: Any) -> Any:
    # Look for capability
    capability = next((t for t in session.agent.tools if t.name == step.tool), None)
    if not capability:
      raise ValueError(f"Tool {step.tool} not found in agent.")

    ctx = RunContext(session)

    # Resolve Ref markers in arguments
    resolved_args = _resolve_refs(step.args, session.completed_steps)

    # Sync RUNNING status
    session.step_statuses[step.id] = StepStatus.RUNNING
    await runner.checkpoint_store.save_step_status(session.session_id, step.id, StepStatus.RUNNING)

    ctx.emit(AgentEvent(
      type=EventType.STEP_STARTED,
      session_id=session.session_id,
      data={"step_id": step.id, "tool": step.tool},
    ))

    # Determine effective retry / timeout policy
    retry_policy = step.retry if step.retry else session.agent.default_retry
    timeout = step.timeout if step.timeout is not None else session.agent.default_timeout

    last_exc: Exception | None = None
    max_attempts = max(1, retry_policy.max_retries + 1)

    for attempt in range(max_attempts):
      try:
        if timeout is not None:
          result = await asyncio.wait_for(
            capability.invoke(ctx, resolved_args),
            timeout=timeout,
          )
        else:
          result = await capability.invoke(ctx, resolved_args)

        # Success — sync state and checkpoint
        session.completed_steps[step.id] = StepResult(data=result)
        session.step_statuses[step.id] = StepStatus.COMPLETED
        await runner.checkpoint_store.save_step_result(session.session_id, step.id, result)

        ctx.emit(AgentEvent(
          type=EventType.STEP_COMPLETED,
          session_id=session.session_id,
          data={"step_id": step.id, "tool": step.tool},
        ))
        return result

      except Exception as exc:
        last_exc = exc
        # On final attempt, record failure and re-raise
        if attempt == max_attempts - 1:
          break
        # Exponential backoff before retry
        await asyncio.sleep(retry_policy.backoff * (2 ** attempt))

    # All attempts exhausted
    failure = StepFailure(
      error_type=type(last_exc).__name__ if last_exc else "UnknownError",
      message=str(last_exc) if last_exc else "Step failed after all retries",
    )
    session.failed_steps[step.id] = failure
    session.step_statuses[step.id] = StepStatus.FAILED
    await runner.checkpoint_store.save_step_error(session.session_id, step.id, last_exc or Exception("Unknown"))

    ctx.emit(AgentEvent(
      type=EventType.STEP_FAILED,
      session_id=session.session_id,
      data={"step_id": step.id, "tool": step.tool, "error": failure.message},
    ))
    raise last_exc or RuntimeError("Step execution failed")

  async def run(self, session: Session, runner: Any) -> RunResult:
    return await self.run_plan(session, runner)

  async def resume(self, session: Session, runner: Any) -> RunResult:
    # DAGScheduler skips already-completed steps via session.completed_steps
    return await self.run_plan(session, runner)


# --------------------------------------------------------------------------- #
# HybridScheduler — Auto-select Conversational vs DAG
# --------------------------------------------------------------------------- #

class HybridScheduler:
  """
  Smart switching between Conversational and DAG scheduler.
  """

  async def run(self, session: Session, runner: Any) -> RunResult:
    if not session.current_plan:
      session.current_plan = await runner._generate_plan(session, "auto")

    groups = session.current_plan.topological_groups()
    if len(groups) <= 1:
      return await ConversationalScheduler().run(session, runner)

    return await DAGScheduler().run_plan(session, runner)