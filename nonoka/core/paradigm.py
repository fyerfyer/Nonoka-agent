import asyncio
import json
import re
from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable

from nonoka.core.session import Session, SessionStatus, StepStatus, StepResult, StepFailure
from nonoka.core.types import RunResult
from nonoka.core.context import RunContext
from nonoka.core.plan import Step, Plan
from nonoka.core.event import AgentEvent, EventType
from nonoka.core.logger import get_logger
from nonoka.core.llm import LLMMessage, LLMMessageRole, LLMResponse
from nonoka.core.memory import MemoryRole
from nonoka.core.errors import (
  ErrorPolicy,
  SafetyError,
  TransientError,
  CancelledError,
  MaxTurnsExceeded,
  MaxStepsExceeded,
  ToolErrorActionType,
)
from nonoka.core.scheduler import _resolve_refs

_logger = get_logger("nonoka.paradigm")


# --------------------------------------------------------------------------- #
# Actor Protocol — anything that can execute a task in a session
# --------------------------------------------------------------------------- #

@runtime_checkable
class Actor(Protocol):
  async def run(self, session: Session, runner: Any, prompt: str = "") -> RunResult:
    ...


# --------------------------------------------------------------------------- #
# ReActAgent — Exploratory paradigm (was ConversationalScheduler)
# --------------------------------------------------------------------------- #

class ReActAgent:
  """
  ReAct loop: LLM re-decides the next action every turn.

  This is the *exploratory* paradigm — suitable for tasks where the path
  is not known upfront (information retrieval, multi-step reasoning,
  dynamic branching).

  Key features:
  * Parallel tool calls within a single turn (bounded by *max_concurrency*).
  * Memory is the primary state carrier.
  * No pre-defined Plan; the conversation context drives execution.

  Args:
    error_policy: How to handle tool errors.
    output_mode: Controls what ``RunResult.data`` contains on success.
      * ``"content"`` (default) — the LLM's final text reply.
      * ``"last_tool_result"`` — the raw result of the last tool call.
    data_extractor: Optional callable ``(Session) -> Any`` that overrides
      ``output_mode`` and extracts custom data from the session.
    max_concurrency: Maximum concurrent tool calls within a single turn.
      Defaults to the framework-wide setting (10).
  """

  def __init__(
    self,
    error_policy: ErrorPolicy | None = None,
    output_mode: str = "content",
    data_extractor: Any | None = None,
    max_concurrency: int | None = None,
  ):
    self.error_policy = error_policy or ErrorPolicy()
    self.output_mode = output_mode
    self.data_extractor = data_extractor
    self.max_concurrency = max_concurrency

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

    # Resolve concurrency limit
    max_concurrency = (
      self.max_concurrency
      if self.max_concurrency is not None
      else session.agent.max_concurrency
    )
    sem = asyncio.Semaphore(max_concurrency)

    try:
      for turn in range(session.agent.max_turns):
        session.check_cancelled()
        session.turn_count = turn + 1

        # Build messages from memory
        messages = self._build_messages(session)

        # Convert tools to OpenAI function schema
        tools = [t.to_json_schema() for t in session.agent.tools] if session.agent.tools else None

        # --- LLM call ------------------------------------------------
        try:
          response = await runner.llm.chat(
            messages=messages,
            tools=tools or None,
          )
        except CancelledError:
          raise
        except Exception as exc:
          _logger.error(
            "llm.chat_failed",
            session_id=session.session_id,
            turn=turn + 1,
            error=str(exc),
          )
          raise TransientError(f"LLM call failed: {exc}") from exc

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
          # Apply output_mode / data_extractor
          final_data = self._extract_result_data(session, parsed_data)
          return RunResult(success=True, data=final_data, session=session)

        # --- Tool calls → add assistant msg, execute tools ---------
        if session.memory is not None:
          await session.memory.add(
            response.content or "",
            MemoryRole.ASSISTANT,
            tool_calls=response.tool_calls,
          )

        # Enforce max_steps before executing tools
        num_tool_calls = len(response.tool_calls)
        if session.agent.max_steps is not None and session.step_count + num_tool_calls > session.agent.max_steps:
          raise MaxStepsExceeded(
            f"Max steps ({session.agent.max_steps}) exceeded for session {session.session_id}"
          )

        # Execute tool calls with bounded concurrency
        async def _execute_limited(tc: dict[str, Any]) -> Any:
          async with sem:
            return await self._execute_tool_call(session, runner, tc)

        tool_results = await asyncio.gather(*[
          _execute_limited(tc)
          for tc in response.tool_calls
        ], return_exceptions=True)

        # Track the last non-exception tool result for output_mode="last_tool_result"
        last_tool_result: Any = None
        for tr in reversed(tool_results):
          if not isinstance(tr, Exception):
            last_tool_result = tr
            break
        session._last_tool_result = last_tool_result  # type: ignore[attr-defined]

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

    except CancelledError as e:
      session.status = SessionStatus.CANCELLED
      session.end_time = __import__("datetime").datetime.now()
      await runner.checkpoint_store.save_session(session.session_id, session.to_state())
      return RunResult(
        success=False,
        session=session,
        error=str(e),
        error_type="cancelled",
      )

    except (MaxTurnsExceeded, MaxStepsExceeded) as e:
      session.status = SessionStatus.FAILED
      session.end_time = __import__("datetime").datetime.now()
      await runner.checkpoint_store.save_session(session.session_id, session.to_state())
      return RunResult(
        success=False,
        session=session,
        error=str(e),
        error_type="limit_exceeded",
      )

    except TransientError as e:
      session.status = SessionStatus.FAILED
      session.end_time = __import__("datetime").datetime.now()
      await runner.checkpoint_store.save_session(session.session_id, session.to_state())
      return RunResult(
        success=False,
        session=session,
        error=str(e),
        error_type="llm_error",
      )

    except Exception as e:
      session.status = SessionStatus.FAILED
      session.end_time = __import__("datetime").datetime.now()
      await runner.checkpoint_store.save_session(session.session_id, session.to_state())
      return RunResult(
        success=False,
        session=session,
        error=str(e),
        error_type="unknown",
      )

  async def resume(self, session: Session, runner: Any) -> RunResult:
    """Resume a conversational session from checkpoint."""
    return await self.run(session, runner, prompt="")

  async def run_stream(
    self,
    session: Session,
    runner: Any,
    prompt: str = "",
  ) -> AsyncIterator[Any]:
    """Streaming variant of the ReAct loop.

    Yields ``StreamEvent`` objects so CLI callers can render LLM output
    incrementally and observe tool-call progress.  The execution semantics
    are identical to ``run()``; only the result delivery is streaming.
    """
    from nonoka.core.runner import StreamEvent

    session.status = SessionStatus.RUNNING
    await runner.checkpoint_store.save_session(session.session_id, session.to_state())

    if prompt and session.memory is not None:
      await session.memory.add(prompt, MemoryRole.USER)

    max_concurrency = (
      self.max_concurrency
      if self.max_concurrency is not None
      else session.agent.max_concurrency
    )
    sem = asyncio.Semaphore(max_concurrency)

    try:
      for turn in range(session.agent.max_turns):
        session.check_cancelled()
        session.turn_count = turn + 1

        messages = self._build_messages(session)
        tools = [t.to_json_schema() for t in session.agent.tools] if session.agent.tools else None

        # --- Streaming LLM call --------------------------------------
        accumulated_content = ""
        accumulated_tool_calls: dict[int, dict[str, Any]] = {}

        try:
          stream = runner.llm.chat_stream(
            messages=messages,
            tools=tools or None,
          )
        except CancelledError:
          raise
        except Exception as exc:
          _logger.error(
            "llm.chat_stream_failed",
            session_id=session.session_id,
            turn=turn + 1,
            error=str(exc),
          )
          raise TransientError(f"LLM streaming call failed: {exc}") from exc

        async for chunk in stream:
          if chunk.content_delta:
            accumulated_content += chunk.content_delta
            yield StreamEvent(
              type="content_delta",
              data={"content": chunk.content_delta},
            )

          if chunk.tool_call_deltas:
            self._accumulate_tool_deltas(accumulated_tool_calls, chunk.tool_call_deltas)

          if chunk.finish_reason:
            break

        tool_calls = self._finalize_tool_calls(accumulated_tool_calls)
        response = LLMResponse(
          content=accumulated_content or None,
          tool_calls=tool_calls or None,
        )

        _logger.info(
          "llm.stream_response",
          session_id=session.session_id,
          turn=turn + 1,
          has_tool_calls=bool(tool_calls),
        )

        # --- No tool calls → final answer ---------------------------
        if not response.tool_calls:
          content = response.content or ""

          if session.memory is not None:
            await session.memory.add(content, MemoryRole.ASSISTANT)

          parsed_data: Any = content
          if session.agent.result_type is not None:
            parsed_data = await self._parse_result_type(session, runner, content, turn)
            if parsed_data is None:
              continue

          session.status = SessionStatus.COMPLETED
          session.end_time = __import__("datetime").datetime.now()
          await runner.checkpoint_store.save_session(session.session_id, session.to_state())
          final_data = self._extract_result_data(session, parsed_data)
          yield StreamEvent(
            type="final",
            data={
              "success": True,
              "data": final_data,
            },
          )
          return

        # --- Tool calls → execute -----------------------------------
        if session.memory is not None:
          await session.memory.add(
            response.content or "",
            MemoryRole.ASSISTANT,
            tool_calls=response.tool_calls,
          )

        yield StreamEvent(
          type="tool_call_start",
          data={"tool_calls": response.tool_calls},
        )

        num_tool_calls = len(response.tool_calls)
        if session.agent.max_steps is not None and session.step_count + num_tool_calls > session.agent.max_steps:
          raise MaxStepsExceeded(
            f"Max steps ({session.agent.max_steps}) exceeded for session {session.session_id}"
          )

        async def _execute_limited(tc: dict[str, Any]) -> tuple[dict[str, Any], Any]:
          async with sem:
            result = await self._execute_tool_call(session, runner, tc)
            return tc, result

        tool_results = await asyncio.gather(*[
          _execute_limited(tc)
          for tc in response.tool_calls
        ], return_exceptions=True)

        last_tool_result: Any = None
        for tr in reversed(tool_results):
          if isinstance(tr, tuple) and not isinstance(tr[1], Exception):
            last_tool_result = tr[1]
            break
        session._last_tool_result = last_tool_result  # type: ignore[attr-defined]

        for item in tool_results:
          if isinstance(item, Exception):
            tc_id = "unknown"
            obs_text = f"Error: {type(item).__name__}: {item}"
          else:
            tc, tr = item
            tc_id = tc.get("id") or tc.get("tool_call_id", "unknown")
            obs_text = json.dumps(tr, ensure_ascii=False, default=str) if not isinstance(tr, str) else tr

            yield StreamEvent(
              type="tool_call_result",
              data={
                "tool_call_id": tc_id,
                "name": tc.get("function", {}).get("name", ""),
                "result_preview": str(tr)[:500],
                "is_error": isinstance(tr, Exception),
              },
            )

          if session.memory is not None:
            await session.memory.add(
              obs_text,
              MemoryRole.TOOL,
              tool_call_id=tc_id,
            )

        await runner.checkpoint_store.save_session(session.session_id, session.to_state())

      raise MaxTurnsExceeded(
        f"Max turns ({session.agent.max_turns}) exceeded for session {session.session_id}"
      )

    except CancelledError as e:
      session.status = SessionStatus.CANCELLED
      session.end_time = __import__("datetime").datetime.now()
      await runner.checkpoint_store.save_session(session.session_id, session.to_state())
      yield StreamEvent(
        type="error",
        data={
          "success": False,
          "error": str(e),
          "error_type": "cancelled",
        },
      )

    except (MaxTurnsExceeded, MaxStepsExceeded) as e:
      session.status = SessionStatus.FAILED
      session.end_time = __import__("datetime").datetime.now()
      await runner.checkpoint_store.save_session(session.session_id, session.to_state())
      yield StreamEvent(
        type="error",
        data={
          "success": False,
          "error": str(e),
          "error_type": "limit_exceeded",
        },
      )

    except TransientError as e:
      session.status = SessionStatus.FAILED
      session.end_time = __import__("datetime").datetime.now()
      await runner.checkpoint_store.save_session(session.session_id, session.to_state())
      yield StreamEvent(
        type="error",
        data={
          "success": False,
          "error": str(e),
          "error_type": "llm_error",
        },
      )

    except Exception as e:
      session.status = SessionStatus.FAILED
      session.end_time = __import__("datetime").datetime.now()
      await runner.checkpoint_store.save_session(session.session_id, session.to_state())
      yield StreamEvent(
        type="error",
        data={
          "success": False,
          "error": str(e),
          "error_type": "unknown",
        },
      )

  # ------------------------------------------------------------------ #
  # Internal helpers
  # ------------------------------------------------------------------ #

  @staticmethod
  def _accumulate_tool_deltas(
    accumulator: dict[int, dict[str, Any]],
    deltas: list[dict[str, Any]],
  ) -> None:
    """Merge incremental tool-call deltas into a complete payload.

    LiteLLM/OpenAI streaming emits partial ``tool_calls`` dicts keyed by
    ``index``.  We accumulate ``id``, ``type``, ``function.name`` and
    ``function.arguments`` across chunks.
    """
    for delta in deltas:
      idx = delta.get("index", 0)
      if idx not in accumulator:
        accumulator[idx] = {"id": None, "type": "function", "function": {"name": "", "arguments": ""}}

      entry = accumulator[idx]
      if delta.get("id"):
        entry["id"] = delta["id"]
      if delta.get("type"):
        entry["type"] = delta["type"]

      func_delta = delta.get("function", {})
      if func_delta:
        current_func = entry["function"]
        if func_delta.get("name"):
          current_func["name"] += func_delta["name"]
        if func_delta.get("arguments"):
          current_func["arguments"] += func_delta["arguments"]

  @staticmethod
  def _finalize_tool_calls(
    accumulator: dict[int, dict[str, Any]],
  ) -> list[dict[str, Any]] | None:
    """Convert accumulated streaming deltas into a complete tool_calls list."""
    if not accumulator:
      return None
    # Sort by index and drop any partial entries that lack a name.
    result = []
    for idx in sorted(accumulator):
      entry = accumulator[idx]
      func = entry.get("function", {})
      if not func.get("name"):
        continue
      result.append({
        "id": entry.get("id") or f"call_{idx}",
        "type": entry.get("type", "function"),
        "function": {
          "name": func.get("name", ""),
          "arguments": func.get("arguments", ""),
        },
      })
    return result or None

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
          raise last_exc
      elif action.type == ToolErrorActionType.REPORT:
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

    data: Any = content
    try:
      data = json.loads(content)
    except json.JSONDecodeError:
      pass

    try:
      if isinstance(data, dict):
        return result_type(**data)
      return result_type(data)
    except (ValidationError, TypeError) as e:
      err_msg = f"Result parsing failed: {e}. Please return valid JSON matching the expected schema."
      if session.memory is not None:
        await session.memory.add(err_msg, MemoryRole.SYSTEM)
      return None

  def _extract_result_data(self, session: Session, parsed_content: Any) -> Any:
    """Apply *output_mode* and *data_extractor* to determine ``RunResult.data``."""
    if self.data_extractor is not None:
      return self.data_extractor(session)

    if self.output_mode == "last_tool_result":
      return getattr(session, "_last_tool_result", None)

    # Default: "content" — return the parsed LLM text reply
    return parsed_content


# --------------------------------------------------------------------------- #
# PlanExecutor — Deterministic orchestration (was DAGScheduler)
# --------------------------------------------------------------------------- #

class PlanExecutor:
  """
  Deterministic plan execution engine.

  This is *not* an Agent paradigm — it is infrastructure.  It takes a
  pre-defined ``Plan`` (DAG) and executes it efficiently:

  * Topological sort → parallel layer execution.
  * ``ref()`` resolution between steps.
  * Per-step retry / timeout / checkpoint.
  * Skips already-completed steps on resume.

  Suitable for: CI/CD pipelines, data ETL, known workflows.
  """

  def __init__(self, max_concurrency: int = 10, error_policy: ErrorPolicy | None = None):
    self.max_concurrency = max_concurrency
    self.error_policy = error_policy or ErrorPolicy()

  async def execute(self, plan: Plan, session: Session, runner: Any) -> RunResult:
    session.status = SessionStatus.RUNNING
    session.current_plan = plan
    await runner.checkpoint_store.save_session(session.session_id, session.to_state())

    if not plan or not plan.steps:
      session.status = SessionStatus.COMPLETED
      await runner.checkpoint_store.save_session(session.session_id, session.to_state())
      return RunResult(success=True, session=session, data=None)

    sem = asyncio.Semaphore(self.max_concurrency)

    async def execute_limited(step: Step):
      async with sem:
        return await self._execute_step(step, session, runner)

    try:
      # Use pre-computed layers instead of calling topological_groups() repeatedly
      for layer_step_ids in plan.layers:
        session.check_cancelled()

        # Handle force_rerun: remove forced steps from completed state
        for sid in layer_step_ids:
          step = plan.get_step(sid)
          if step and step.force_rerun:
            session.completed_steps.pop(sid, None)
            session.step_statuses.pop(sid, None)
            session.failed_steps.pop(sid, None)

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
          session.end_time = __import__("datetime").datetime.now()
          await runner.checkpoint_store.save_session(session.session_id, session.to_state())
          first_fail = failures[0]
          return RunResult(
            success=False,
            session=session,
            error=f"Step(s) failed during execution: {first_fail}",
            error_type="tool_error",
          )

        # Checkpoint layer
        await runner.checkpoint_store.save_session(session.session_id, session.to_state())

      session.status = SessionStatus.COMPLETED
      session.end_time = __import__("datetime").datetime.now()
      await runner.checkpoint_store.save_session(session.session_id, session.to_state())
      # Collect final output from the last executed step
      final_data = self._extract_final_data(plan, session)
      return RunResult(success=True, session=session, data=final_data)

    except CancelledError as e:
      session.status = SessionStatus.CANCELLED
      session.end_time = __import__("datetime").datetime.now()
      await runner.checkpoint_store.save_session(session.session_id, session.to_state())
      return RunResult(
        success=False,
        session=session,
        error=str(e),
        error_type="cancelled",
      )

    except MaxStepsExceeded as e:
      session.status = SessionStatus.FAILED
      session.end_time = __import__("datetime").datetime.now()
      await runner.checkpoint_store.save_session(session.session_id, session.to_state())
      return RunResult(
        success=False,
        session=session,
        error=str(e),
        error_type="limit_exceeded",
      )

    except Exception as e:
      session.status = SessionStatus.FAILED
      session.end_time = __import__("datetime").datetime.now()
      await runner.checkpoint_store.save_session(session.session_id, session.to_state())
      return RunResult(
        success=False,
        session=session,
        error=str(e),
        error_type="unknown",
      )

  async def _execute_step(self, step: Step, session: Session, runner: Any) -> Any:
    """Execute a single step with retry / timeout / checkpoint."""
    capability = next((t for t in session.agent.tools if t.name == step.tool), None)
    if not capability:
      raise ValueError(f"Tool {step.tool} not found in agent.")

    ctx = RunContext(session)

    # Enforce max_steps (total step execution budget)
    if session.agent.max_steps is not None and session.step_count >= session.agent.max_steps:
      raise MaxStepsExceeded(
        f"Max steps ({session.agent.max_steps}) exceeded for session {session.session_id}"
      )
    session.step_count += 1

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
        if attempt == max_attempts - 1:
          break
        await asyncio.sleep(retry_policy.backoff * (2 ** attempt))

    # All attempts exhausted — consult ErrorPolicy for final disposition
    last_exc = last_exc or RuntimeError("Step execution failed")
    action = self.error_policy.on_tool_error(last_exc, step.id)

    if action.type == ToolErrorActionType.REPORT:
      # Report error as step result and continue (don't fail the plan)
      error_result = {"error": f"{type(last_exc).__name__}: {last_exc}"}
      session.completed_steps[step.id] = StepResult(data=error_result)
      session.step_statuses[step.id] = StepStatus.COMPLETED
      await runner.checkpoint_store.save_step_result(session.session_id, step.id, error_result)
      ctx.emit(AgentEvent(
        type=EventType.STEP_COMPLETED,
        session_id=session.session_id,
        data={"step_id": step.id, "tool": step.tool, "error_reported": True},
      ))
      return error_result

    if action.type == ToolErrorActionType.HALT:
      raise SafetyError(f"Halted on step error: {last_exc}") from last_exc

    # FAIL (default) — raise the exception to fail the plan
    failure = StepFailure(
      error_type=type(last_exc).__name__,
      message=str(last_exc),
    )
    session.failed_steps[step.id] = failure
    session.step_statuses[step.id] = StepStatus.FAILED
    await runner.checkpoint_store.save_step_error(session.session_id, step.id, last_exc)

    ctx.emit(AgentEvent(
      type=EventType.STEP_FAILED,
      session_id=session.session_id,
      data={"step_id": step.id, "tool": step.tool, "error": failure.message},
    ))
    raise last_exc

  def _extract_final_data(self, plan: Plan, session: Session) -> Any:
    """Extract the final output from a completed plan.

    Returns the data of the last step in topological order that has a
    completed result.  If no steps completed, returns ``None``.
    """
    if not plan.steps:
      return None
    # Use pre-computed layers instead of calling topological_groups() again
    for layer in reversed(plan.layers):
      for step_id in reversed(layer):
        if step_id in session.completed_steps:
          return session.completed_steps[step_id].data
    return None

  async def resume(self, plan: Plan, session: Session, runner: Any) -> RunResult:
    """Resume plan execution from checkpoint (skips completed steps)."""
    return await self.execute(plan, session, runner)


# --------------------------------------------------------------------------- #
# EvaluationResult — output of an Evaluator
# --------------------------------------------------------------------------- #

class EvaluationResult:
  """Result of evaluating an execution attempt."""

  def __init__(
    self,
    passed: bool,
    feedback: str = "",
    score: float | None = None,
  ):
    self.passed = passed
    self.feedback = feedback
    self.score = score


# --------------------------------------------------------------------------- #
# Evaluator Protocol
# --------------------------------------------------------------------------- #

@runtime_checkable
class Evaluator(Protocol):
  """Protocol for quality evaluators used by ReflectiveAgent."""

  async def evaluate(self, result: RunResult) -> EvaluationResult:
    ...


# --------------------------------------------------------------------------- #
# ReflectiveAgent — Quality-driven paradigm
# --------------------------------------------------------------------------- #

class ReflectiveAgent:
  """
  Quality-driven execution paradigm: Actor → Evaluate → Revise (loop).

  The *Actor* performs one round of task execution.  The *Evaluator*
  assesses the result quality.  If it does not pass, the feedback is fed
  back into the Actor's context for another attempt.

  Actor is pluggable — it can be a ``ReActAgent`` or a ``PlanExecutor``
  wrapped in a thin adapter.

  Usage::

    actor = ReActAgent()
    evaluator = ToolEvaluator(validate_config)
    reflective = ReflectiveAgent(actor=actor, evaluator=evaluator, max_iterations=5)
    result = await reflective.run(session, runner, prompt="Fix config files")
  """

  def __init__(
    self,
    actor: Actor,
    evaluator: Evaluator,
    max_iterations: int = 3,
  ):
    self.actor = actor
    self.evaluator = evaluator
    self.max_iterations = max_iterations

  async def run(
    self,
    session: Session,
    runner: Any,
    prompt: str = "",
  ) -> RunResult:
    """Execute the reflective loop."""
    session.status = SessionStatus.RUNNING
    await runner.checkpoint_store.save_session(session.session_id, session.to_state())

    feedback = ""
    best_result: RunResult | None = None
    best_score: float = -1.0

    for iteration in range(1, self.max_iterations + 1):
      session.check_cancelled()

      _logger.info(
        "reflective.iteration_start",
        session_id=session.session_id,
        iteration=iteration,
        max_iterations=self.max_iterations,
      )

      # Inject feedback into the prompt for subsequent iterations
      effective_prompt = prompt
      if feedback and iteration > 1:
        effective_prompt = (
          f"{prompt}\n\n"
          f"[Previous attempt feedback — please address these issues]:\n"
          f"{feedback}"
        )

      # Actor executes
      result = await self.actor.run(session, runner, prompt=effective_prompt)

      # Short-circuit: if actor failed catastrophically, don't bother evaluating
      if not result.success and result.error:
        _logger.warning(
          "reflective.actor_failed",
          session_id=session.session_id,
          iteration=iteration,
          error=result.error,
        )
        # Keep the last failure if nothing better exists
        if best_result is None:
          best_result = result
        continue

      # Evaluate the result
      eval_result = await self.evaluator.evaluate(result)

      _logger.info(
        "reflective.evaluation",
        session_id=session.session_id,
        iteration=iteration,
        passed=eval_result.passed,
        score=eval_result.score,
      )

      # Track best attempt by score (if available)
      score = eval_result.score if eval_result.score is not None else (1.0 if eval_result.passed else 0.0)
      if score > best_score:
        best_score = score
        best_result = result

      if eval_result.passed:
        session.status = SessionStatus.COMPLETED
        session.end_time = __import__("datetime").datetime.now()
        await runner.checkpoint_store.save_session(session.session_id, session.to_state())
        return RunResult(
          success=True,
          data=result.data,
          session=session,
        )

      # Prepare feedback for next iteration
      feedback = eval_result.feedback
      if not feedback:
        feedback = "The previous attempt did not meet quality standards. Please try a different approach."

    # All iterations exhausted — return the best attempt we have
    _logger.warning(
      "reflective.max_iterations_reached",
      session_id=session.session_id,
      max_iterations=self.max_iterations,
      best_score=best_score,
    )
    session.status = SessionStatus.FAILED
    session.end_time = __import__("datetime").datetime.now()
    await runner.checkpoint_store.save_session(session.session_id, session.to_state())
    return RunResult(
      success=False,
      data=best_result.data if best_result else None,
      session=session,
      error=f"Max reflective iterations ({self.max_iterations}) reached. "
        f"Best score: {best_score}. Last feedback: {feedback}",
      error_type="limit_exceeded",
    )


# --------------------------------------------------------------------------- #
# Convenience: wrap a plain tool/callable as an Evaluator
# --------------------------------------------------------------------------- #

class ToolEvaluator:
  """
  Wrap a deterministic validation tool as an ``Evaluator``.

  The tool should accept a ``RunContext`` plus whatever fields it needs,
  and return a dict like ``{"passed": bool, "feedback": str}``.

  Usage::

    @tool
    async def validate_config(ctx: RunContext, config: str) -> dict:
      errors = lint(config)
      return {"passed": len(errors) == 0, "feedback": "\n".join(errors)}

    evaluator = ToolEvaluator(validate_config, data_extractor=lambda r: r.data)
  """

  def __init__(
    self,
    validate_tool: Any,
    data_extractor: Any | None = None,
  ):
    self.validate_tool = validate_tool
    self.data_extractor = data_extractor or (lambda r: r.data)

  async def evaluate(self, result: RunResult) -> EvaluationResult:
    from nonoka.core.context import RunContext

    data = self.data_extractor(result)
    session = result.session
    if session is None:
      return EvaluationResult(
        passed=False,
        feedback="No session available for evaluation.",
      )

    # Build a synthetic RunContext from the Session
    run_ctx = RunContext(session)

    try:
      raw = await self.validate_tool.invoke(run_ctx, {"data": data})
    except Exception as exc:
      return EvaluationResult(
        passed=False,
        feedback=f"Evaluation tool failed: {type(exc).__name__}: {exc}",
      )

    if isinstance(raw, dict):
      passed = raw.get("passed", False)
      feedback = raw.get("feedback", "")
      score = raw.get("score")
      return EvaluationResult(passed=passed, feedback=feedback, score=score)

    # Treat truthy return as passed
    return EvaluationResult(passed=bool(raw), feedback=str(raw))
