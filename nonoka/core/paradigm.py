from __future__ import annotations

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
  ToolFatalError,
  TransientError,
  CancelledError,
  MaxTurnsExceeded,
  MaxStepsExceeded,
  ToolErrorActionType,
)
from nonoka.core.scheduler import _resolve_refs
from nonoka.core.hooks import HookContext
from nonoka.core.tool_response import unwrap_tool_response

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
    max_repeated_tool_calls: int = 3,
    loop_similarity_threshold: int = 3,
  ):
    self.error_policy = error_policy or ErrorPolicy()
    self.output_mode = output_mode
    self.data_extractor = data_extractor
    self.max_concurrency = max_concurrency
    # Loop detection configuration
    self.max_repeated_tool_calls = max_repeated_tool_calls
    self.loop_similarity_threshold = loop_similarity_threshold

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

        # Build messages from memory (or directly from prompt if no memory)
        if session.memory is not None:
          messages = self._build_messages(session)
        else:
          messages = []
          if session.agent.system_prompt:
            messages.append(LLMMessage(role=LLMMessageRole.SYSTEM, content=session.agent.system_prompt))
          if prompt:
            messages.append(LLMMessage(role=LLMMessageRole.USER, content=prompt))

        # Convert tools to OpenAI function schema
        # Filter out temporarily blocked tools (loop detection escalation)
        blocked = getattr(session, "_blocked_tools", set())
        available_tools = [
          t for t in session.agent.tools
          if t.name not in blocked
        ] if session.agent.tools else []
        tools = [t.to_json_schema() for t in available_tools] if available_tools else None

        # --- LLM call ------------------------------------------------
        hook_ctx = HookContext(session=session, runner=runner)
        await runner.hooks.emit_llm_request(hook_ctx, messages, tools)
        try:
          response = await runner.llm.chat(
            messages=messages,
            tools=tools or None,
          )
          await runner.hooks.emit_llm_response(hook_ctx, response)
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

        # Check for fatal errors (HALT / FAIL) before adding to memory.
        # asyncio.gather(return_exceptions=True) swallows exceptions; we must
        # re-raise or translate them into a terminal RunResult here.
        # Only SafetyError (HALT) and ToolFatalError (FAIL) terminate the loop;
        # other exceptions (e.g. ValueError from a missing tool) are still
        # fed back to the LLM as observations so it can self-correct.
        for tr in tool_results:
          if isinstance(tr, (SafetyError, ToolFatalError)):
            session.status = SessionStatus.FAILED
            session.end_time = __import__("datetime").datetime.now()
            await runner.checkpoint_store.save_session(session.session_id, session.to_state())
            error_type = "halted" if isinstance(tr, SafetyError) else "tool_error"
            return RunResult(
              success=False,
              session=session,
              error=str(tr),
              error_type=error_type,
            )
          if isinstance(tr, asyncio.CancelledError):
            raise tr

        # Track the last non-exception tool result for output_mode="last_tool_result"
        last_tool_result: Any = None
        for tr in reversed(tool_results):
          if not isinstance(tr, Exception):
            last_tool_result = tr
            break
        session._last_tool_result = last_tool_result  # type: ignore[attr-defined]

        # Add tool observations to memory.  SYSTEM guidance is injected
        # *after* all TOOL entries so that ASSISTANT+tool_calls messages stay
        # contiguous with their corresponding TOOL responses (required by
        # OpenAI / DeepSeek API).
        tool_guidance: list[str] = []
        has_more_notices: list[str] = []
        for tc, tr in zip(response.tool_calls, tool_results):
          tc_id = tc.get("id") or tc.get("tool_call_id", "unknown")
          func_name = tc.get("function", {}).get("name", "")
          if isinstance(tr, Exception):
            obs_text = f"Error: {type(tr).__name__}: {tr}"
          else:
            obs_text = json.dumps(tr, ensure_ascii=False, default=str) if not isinstance(tr, str) else tr

          if session.memory is not None:
            await session.memory.add(
              obs_text,
              MemoryRole.TOOL,
              tool_call_id=tc_id,
              tool_name=func_name,
            )

            # Collect ToolResponse metadata to inject after all tool messages
            if isinstance(tr, dict):
              suggested = tr.get("suggested_next_step")
              if suggested:
                tool_guidance.append(f"[Tool guidance] {suggested}")

              if tr.get("has_more") is False:
                has_more_notices.append(
                  f"[System notice] {func_name or 'the tool'} returned 'has_more': false — "
                  "there is no additional data available."
                )

        if session.memory is not None:
          for notice in has_more_notices + tool_guidance:
            await session.memory.add(notice, MemoryRole.SYSTEM)

        # --- Loop detection --------------------------------------------
        should_terminate = await self._detect_and_break_loops(
          session, response.tool_calls, tool_results
        )
        if should_terminate:
          session.status = SessionStatus.FAILED
          session.end_time = __import__("datetime").datetime.now()
          await runner.checkpoint_store.save_session(session.session_id, session.to_state())
          return RunResult(
            success=False,
            session=session,
            error=f"Agent loop detected: tool '{session._tool_call_history[-1][0] if hasattr(session, '_tool_call_history') else 'unknown'}' was called repeatedly without meaningful progress.",
            error_type="loop_detected",
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

        # Build messages from memory (or directly from prompt if no memory)
        if session.memory is not None:
          messages = self._build_messages(session)
        else:
          messages = []
          if session.agent.system_prompt:
            messages.append(LLMMessage(role=LLMMessageRole.SYSTEM, content=session.agent.system_prompt))
          if prompt:
            messages.append(LLMMessage(role=LLMMessageRole.USER, content=prompt))

        # Filter out temporarily blocked tools (loop detection escalation)
        blocked = getattr(session, "_blocked_tools", set())
        available_tools = [
          t for t in session.agent.tools
          if t.name not in blocked
        ] if session.agent.tools else []
        tools = [t.to_json_schema() for t in available_tools] if available_tools else None

        # --- Streaming LLM call --------------------------------------
        accumulated_content = ""
        accumulated_tool_calls: dict[int, dict[str, Any]] = {}

        # Hook: llm request (streaming)
        hook_ctx = HookContext(session=session, runner=runner)
        await runner.hooks.emit_llm_request(hook_ctx, messages, tools)

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

        # Hook: llm response (streaming)
        await runner.hooks.emit_llm_response(hook_ctx, response)

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

        # Check for fatal errors (HALT / FAIL) before streaming results.
        # Only SafetyError (HALT) and ToolFatalError (FAIL) terminate;
        # other exceptions are still streamed as tool_call_result events.
        for item in tool_results:
          if isinstance(item, (SafetyError, ToolFatalError)):
            session.status = SessionStatus.FAILED
            session.end_time = __import__("datetime").datetime.now()
            await runner.checkpoint_store.save_session(session.session_id, session.to_state())
            error_type = "halted" if isinstance(item, SafetyError) else "tool_error"
            yield StreamEvent(
              type="error",
              data={
                "success": False,
                "error": str(item),
                "error_type": error_type,
              },
            )
            return
          if isinstance(item, asyncio.CancelledError):
            raise item

        last_tool_result: Any = None
        for tr in reversed(tool_results):
          if isinstance(tr, tuple) and not isinstance(tr[1], Exception):
            last_tool_result = tr[1]
            break
        session._last_tool_result = last_tool_result  # type: ignore[attr-defined]

        for item in tool_results:
          if isinstance(item, Exception):
            # Non-fatal exception (e.g. ValueError from missing tool) —
            # stream as an error result so the LLM can self-correct.
            obs_text = f"Error: {type(item).__name__}: {item}"
            yield StreamEvent(
              type="tool_call_result",
              data={
                "tool_call_id": "unknown",
                "name": "",
                "result_preview": obs_text[:500],
                "is_error": True,
              },
            )
            if session.memory is not None:
              await session.memory.add(
                obs_text,
                MemoryRole.TOOL,
                tool_call_id="unknown",
                tool_name="",
              )
            continue

          tc, tr = item
          tc_id = tc.get("id") or tc.get("tool_call_id", "unknown")
          tc_name = tc.get("function", {}).get("name", "")
          obs_text = json.dumps(tr, ensure_ascii=False, default=str) if not isinstance(tr, str) else tr

          yield StreamEvent(
            type="tool_call_result",
            data={
              "tool_call_id": tc_id,
              "name": tc_name,
              "result_preview": str(tr)[:500],
              "is_error": False,
            },
          )

          if session.memory is not None:
            await session.memory.add(
              obs_text,
              MemoryRole.TOOL,
              tool_call_id=tc_id,
              tool_name=tc_name,
            )

        # Inject ToolResponse metadata SYSTEM messages *after* all TOOL
        # entries so that ASSISTANT+tool_calls stay contiguous with their
        # corresponding TOOL responses.
        if session.memory is not None:
          stream_guidance: list[str] = []
          stream_notices: list[str] = []
          for item in tool_results:
            if isinstance(item, Exception):
              continue
            tc, tr = item
            if isinstance(tr, dict):
              suggested = tr.get("suggested_next_step")
              if suggested:
                stream_guidance.append(f"[Tool guidance] {suggested}")
              if tr.get("has_more") is False:
                func_name = tc.get("function", {}).get("name", "the tool")
                stream_notices.append(
                  f"[System notice] {func_name} returned 'has_more': false — "
                  "there is no additional data available."
                )
          for notice in stream_notices + stream_guidance:
            await session.memory.add(notice, MemoryRole.SYSTEM)

        # --- Loop detection (streaming) --------------------------------
        should_terminate = await self._detect_and_break_loops(
          session, response.tool_calls, tool_results
        )
        if should_terminate:
          session.status = SessionStatus.FAILED
          session.end_time = __import__("datetime").datetime.now()
          await runner.checkpoint_store.save_session(session.session_id, session.to_state())
          yield StreamEvent(
            type="error",
            data={
              "success": False,
              "error": f"Agent loop detected: tool was called repeatedly without meaningful progress.",
              "error_type": "loop_detected",
            },
          )
          return

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

    # Hook: tool start
    hook_ctx = HookContext(session=session, runner=runner)
    await runner.hooks.emit_tool_start(hook_ctx, name, arguments)

    result: Any = None
    error: Exception | None = None
    try:
      result = await capability.invoke(ctx, arguments)
    except Exception as exc:
      error = exc
      action = self.error_policy.on_tool_error(exc, f"{name}_turn_{session.turn_count}")

      if action.type == ToolErrorActionType.RETRY:
        max_retries = max(1, action.kwargs.get("max_retries", 3))
        last_exc = exc
        for attempt in range(max_retries):
          try:
            result = await capability.invoke(ctx, arguments)
            error = None
            break
          except Exception as retry_exc:
            last_exc = retry_exc
            error = last_exc
            if attempt == max_retries - 1:
              raise last_exc
        else:
          raise last_exc
      elif action.type == ToolErrorActionType.REPORT:
        result = {"error": f"{type(exc).__name__}: {exc}"}
        error = None
      elif action.type == ToolErrorActionType.HALT:
        raise SafetyError(f"Halted on tool error: {exc}") from exc
      else:
        # FAIL — wrap in ToolFatalError so the ReAct loop knows to terminate
        # rather than feed the error back to the LLM as an observation.
        raise ToolFatalError(f"Tool execution failed: {exc}") from exc

    # Hook: tool end
    await runner.hooks.emit_tool_end(hook_ctx, name, arguments, result, error)

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

  # ------------------------------------------------------------------ #
  # Loop detection
  # ------------------------------------------------------------------ #

  async def _detect_and_break_loops(
    self,
    session: Session,
    tool_calls: list[dict[str, Any]],
    tool_results: list[Any],
  ) -> bool:
    """Detect repetitive tool-call patterns and take graded action.

    Returns ``True`` when a terminal loop has been detected and the caller
    should abort the turn (after saving checkpoint).

    Detection layers (in order):
    1. **has_more exemption** – calls that legitimately paginate are not
       counted toward loop thresholds.
    2. **Consecutive tool** – same tool called repeatedly (even with
       different arguments).
    3. **Identical arguments** – same (tool_name, arguments) pair appears
       multiple times in the recent window.
    4. **Short-cycle** – A→B→A→B or A→B→C→A→B→C patterns.
    5. **Result similarity** – same tool produces substantively identical
       output across consecutive calls (catches ``grep(\"foo\")`` →
       ``grep(\"Foo\")`` where results are the same).

    Response escalation:
    | trigger count | action |
    |---------------|--------|
    | 1st           | inject warning into memory |
    | 2nd           | stronger warning + temporarily block the tool(s) |
    | 3rd+          | force termination (return ``True``) |
    """
    if session.memory is None:
      return False

    # Build a signature for each tool call in this turn
    current_sigs: list[tuple[str, str]] = []
    for tc in tool_calls:
      func = tc.get("function", {})
      name = func.get("name", "")
      args = func.get("arguments", "")
      args_norm = re.sub(r"\s+", "", str(args))
      current_sigs.append((name, args_norm))

    # -- History tracking -------------------------------------------------
    if not hasattr(session, "_tool_call_history"):
      session._tool_call_history = []  # type: ignore[attr-defined]
      session._tool_result_history = []  # type: ignore[attr-defined]
      session._loop_trigger_count = 0  # type: ignore[attr-defined]
      session._blocked_tools = set()  # type: ignore[attr-defined]

    session._tool_call_history.extend(current_sigs)  # type: ignore[attr-defined]
    session._tool_result_history.extend(tool_results)  # type: ignore[attr-defined]
    history: list[tuple[str, str]] = session._tool_call_history  # type: ignore[attr-defined]
    result_history: list[Any] = session._tool_result_history  # type: ignore[attr-defined]

    # -- Heuristic helpers ------------------------------------------------
    def _has_more_true(idx: int) -> bool:
      """Check whether the result at *idx* indicates more data is available."""
      if idx < 0 or idx >= len(result_history):
        return False
      res = result_history[idx]
      return isinstance(res, dict) and res.get("has_more") is True

    def _has_more_false(idx: int) -> bool:
      res = result_history[idx] if 0 <= idx < len(result_history) else None
      return isinstance(res, dict) and res.get("has_more") is False

    def _results_similar(a: Any, b: Any, threshold: float = 0.9) -> bool:
      """Compare two tool results for substantive equality."""
      text_a = json.dumps(a, sort_keys=True, ensure_ascii=False, default=str)
      text_b = json.dumps(b, sort_keys=True, ensure_ascii=False, default=str)
      if len(text_a) < 100 and len(text_b) < 100:
        return text_a == text_b
      import difflib
      return difflib.SequenceMatcher(None, text_a, text_b).ratio() > threshold

    def _is_error_result(res: Any) -> bool:
      """Check whether a tool result indicates failure (raw Exception or
      REPORT-policy dict with an ``error`` key)."""
      if isinstance(res, Exception):
        return True
      if isinstance(res, dict) and "error" in res:
        return True
      return False

    # -- Heuristic 1: consecutive same tool (with has_more awareness) -----
    consecutive_count = 1
    for i in range(len(history) - 2, -1, -1):
      if history[i][0] == history[-1][0]:
        prev_idx = len(result_history) - (len(history) - i)
        # Repair-attempt exemption: if the previous call failed (result is an
        # error/Exception) and arguments are different, the LLM is likely
        # correcting its mistake — stop counting and do not treat this as a loop.
        prev_result = result_history[prev_idx] if 0 <= prev_idx < len(result_history) else None
        is_repair_attempt = (
          _is_error_result(prev_result) and history[i][1] != history[-1][1]
        )
        if is_repair_attempt:
          break
        # If the *previous* call's result said has_more=True, this call is
        # likely a legitimate pagination request — count it at half weight.
        if _has_more_true(prev_idx):
          consecutive_count += 0.5
        else:
          consecutive_count += 1
      else:
        break

    # Accelerate detection when has_more=false but tool is still called
    h1_threshold = self.max_repeated_tool_calls
    if history and _has_more_false(len(result_history) - 1):
      h1_threshold = max(2, h1_threshold - 1)

    h1_triggered = consecutive_count >= h1_threshold

    # -- Heuristic 2: repeated (name, args) pair --------------------------
    recent = history[-10:]
    repeat_counts: dict[tuple[str, str], int] = {}
    for sig in recent:
      repeat_counts[sig] = repeat_counts.get(sig, 0) + 1
    max_repeat = max(repeat_counts.values()) if repeat_counts else 0
    h2_triggered = max_repeat >= self.loop_similarity_threshold

    # -- Heuristic 3: short-cycle detection (A→B→A→B, A→B→C→A→B→C) -----
    h3_triggered = False
    if len(history) >= 4:
      names = [s[0] for s in history[-4:]]
      if names[0] == names[2] and names[1] == names[3] and names[0] != names[1]:
        h3_triggered = True
    if len(history) >= 6 and not h3_triggered:
      names = [s[0] for s in history[-6:]]
      if names[:3] == names[3:]:
        h3_triggered = True

    # -- Heuristic 4: result similarity -----------------------------------
    h4_triggered = False
    if len(result_history) >= 3 and len(history) >= 3:
      # Compare last 3 results of the same tool
      last_tool = history[-1][0]
      same_tool_results = [
        result_history[i]
        for i in range(len(history))
        if history[i][0] == last_tool
      ][-3:]
      if len(same_tool_results) == 3:
        if (
          _results_similar(same_tool_results[0], same_tool_results[1])
          and _results_similar(same_tool_results[1], same_tool_results[2])
        ):
          h4_triggered = True

    loop_detected = h1_triggered or h2_triggered or h3_triggered or h4_triggered

    if not loop_detected:
      return False

    # -- Escalation -------------------------------------------------------
    session._loop_trigger_count += 1  # type: ignore[attr-defined]
    trigger_count: int = session._loop_trigger_count  # type: ignore[attr-defined]

    tool_name = history[-1][0] if history else "unknown"
    _logger.warning(
      "react.loop_detected",
      session_id=session.session_id,
      trigger_count=trigger_count,
      h1=h1_triggered,
      h2=h2_triggered,
      h3=h3_triggered,
      h4=h4_triggered,
      tool=tool_name,
    )

    if trigger_count == 1:
      warning = (
        f"[System notice] The tool '{tool_name}' has been called repeatedly "
        "with similar arguments or patterns. Please STOP calling this tool "
        "and proceed based on the information you already have."
      )
      await session.memory.add(warning, MemoryRole.SYSTEM)
      return False

    if trigger_count == 2:
      warning = (
        f"[System notice] Loop confirmed — '{tool_name}' is still being called "
        "repeatedly. This tool is now TEMPORARILY DISABLED. You must use "
        "other tools or provide a final answer."
      )
      await session.memory.add(warning, MemoryRole.SYSTEM)
      # Block the tool(s) involved in the loop
      for sig in recent:
        if repeat_counts.get(sig, 0) >= 2:
          session._blocked_tools.add(sig[0])  # type: ignore[attr-defined]
      if tool_name not in session._blocked_tools:  # type: ignore[attr-defined]
        session._blocked_tools.add(tool_name)  # type: ignore[attr-defined]
      return False

    # trigger_count >= 3 — force termination
    warning = (
      f"[System notice] Agent loop detected after multiple warnings. "
      f"Tool '{tool_name}' was called repeatedly without meaningful progress. "
      "Terminating execution."
    )
    await session.memory.add(warning, MemoryRole.SYSTEM)
    return True


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

    # Hook: plan start
    hook_ctx = HookContext(session=session, runner=runner)
    await runner.hooks.emit_plan_start(hook_ctx)

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

    # Hook: plan step start
    hook_ctx = HookContext(session=session, runner=runner)
    await runner.hooks.emit_plan_step_start(hook_ctx, step.id, step.tool, resolved_args)

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

        # Hook: plan step end (success)
        await runner.hooks.emit_plan_step_end(hook_ctx, step.id, step.tool, result, None)

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

      # Hook: plan step end (error reported)
      await runner.hooks.emit_plan_step_end(hook_ctx, step.id, step.tool, error_result, last_exc)

      ctx.emit(AgentEvent(
        type=EventType.STEP_COMPLETED,
        session_id=session.session_id,
        data={"step_id": step.id, "tool": step.tool, "error_reported": True},
      ))
      return error_result

    if action.type == ToolErrorActionType.HALT:
      # Hook: plan step end (halted)
      await runner.hooks.emit_plan_step_end(hook_ctx, step.id, step.tool, None, last_exc)
      raise SafetyError(f"Halted on step error: {last_exc}") from last_exc

    # FAIL (default) — raise the exception to fail the plan
    failure = StepFailure(
      error_type=type(last_exc).__name__,
      message=str(last_exc),
    )
    session.failed_steps[step.id] = failure
    session.step_statuses[step.id] = StepStatus.FAILED
    await runner.checkpoint_store.save_step_error(session.session_id, step.id, last_exc)

    # Hook: plan step end (failed)
    await runner.hooks.emit_plan_step_end(hook_ctx, step.id, step.tool, None, last_exc)

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

    # Unwrap normalised tool-response wrapper if present
    eval_dict = raw
    if isinstance(raw, dict) and "result" in raw and "has_more" in raw:
      eval_dict = raw.get("result", raw)

    if isinstance(eval_dict, dict):
      passed = eval_dict.get("passed", False)
      feedback = eval_dict.get("feedback", "")
      score = eval_dict.get("score")
      return EvaluationResult(passed=passed, feedback=feedback, score=score)

    # Treat truthy return as passed
    return EvaluationResult(passed=bool(eval_dict), feedback=str(eval_dict))
