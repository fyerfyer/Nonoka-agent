from __future__ import annotations

import asyncio
import json
import re
import anyio
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
  ApprovalRequiredError,
  ErrorPolicy,
  ExternalToolExecutionRequiredError,
  SafetyError,
  ToolFatalError,
  TransientError,
  CancelledError,
  RuntimeTerminatedError,
  MaxTurnsExceeded,
  MaxStepsExceeded,
  ToolErrorActionType,
)
from nonoka.core.scheduler import _resolve_refs
from nonoka.core.hooks import HookContext
from nonoka.core.execution import ToolExecutionCoordinator, execution_for
from nonoka.core.extensions import LoopExtension, LoopExtensionContext, LoopExtensionManager

_logger = get_logger("nonoka.paradigm")

_TOOL_CALL_PROGRESS_INTERVAL_CHARS = 1024
_TOOL_CALL_MARKUP_RE = re.compile(
  r"(?:DSML.{0,32}(?:tool_calls|invoke)|<tool_call|<function_calls)",
  re.IGNORECASE | re.DOTALL,
)


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
    extensions: list[LoopExtension] | None = None,
  ):
    self.error_policy = error_policy or ErrorPolicy()
    self.output_mode = output_mode
    self.data_extractor = data_extractor
    self.max_concurrency = max_concurrency
    # Loop detection configuration
    self.max_repeated_tool_calls = max_repeated_tool_calls
    self.loop_similarity_threshold = loop_similarity_threshold
    self.extensions = list(extensions or [])

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
    extension_manager = LoopExtensionManager([
      *list(getattr(session.agent, "extensions", [])), *self.extensions,
    ])
    try:
      while True:
        turn = session.begin_model_turn() - 1
        await session.enforce_context_budget()

        start_decision = await extension_manager.before_turn(LoopExtensionContext(
          session=session, runner=runner, prompt=prompt, turn=turn + 1,
        ))
        if start_decision.failure:
          return await self._extension_failure(session, runner, start_decision.failure)
        if start_decision.feedback and session.memory is not None:
          await session.memory.add(start_decision.feedback, MemoryRole.SYSTEM)

        # Build messages from memory (or directly from prompt if no memory)
        if session.memory is not None:
          messages = (
            self._build_tool_free_finalization_messages(session, start_decision.feedback)
            if start_decision.disable_tools
            else self._build_messages(session)
          )
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
        if start_decision.disable_tools:
          available_tools = []
        tools = [t.to_json_schema() for t in available_tools] if available_tools else None

        # --- LLM call ------------------------------------------------
        hook_ctx = HookContext(session=session, runner=runner)
        session.trace.record_turn_request(
          turn + 1,
          [message.model_dump(exclude_none=True) for message in messages],
          tools=tools, temperature=session.agent.temperature,
          max_tokens=session.agent.max_tokens,
        )
        await runner.hooks.emit_llm_request(hook_ctx, messages, tools)
        try:
          model_timeout = session.runtime_state.limits.model_timeout_seconds
          remaining = session.runtime_state.remaining_seconds()
          effective_timeout = min(
            value for value in (model_timeout, remaining) if value is not None
          ) if model_timeout is not None or remaining is not None else None
          if effective_timeout is None:
            complete = getattr(type(runner), "complete", None)
            response = await (runner.complete(
              messages=messages, tools=tools or None,
              temperature=session.agent.temperature, max_tokens=session.agent.max_tokens,
            ) if complete is not None else runner.llm.chat(
              messages=messages, tools=tools or None,
              temperature=session.agent.temperature, max_tokens=session.agent.max_tokens,
            ))
          else:
            with anyio.fail_after(effective_timeout):
              complete = getattr(type(runner), "complete", None)
              response = await (runner.complete(
                messages=messages, tools=tools or None,
                temperature=session.agent.temperature, max_tokens=session.agent.max_tokens,
              ) if complete is not None else runner.llm.chat(
                messages=messages, tools=tools or None,
                temperature=session.agent.temperature, max_tokens=session.agent.max_tokens,
              ))
          await runner.hooks.emit_llm_response(hook_ctx, response)
          if getattr(type(runner), "record_llm_usage", None) is not None:
            await runner.record_llm_usage(session, response.usage, cache_hit=bool(response.usage.pop("_cache_hit", False)))
        except TimeoutError as exc:
          deadline_limited = remaining is not None and (
            model_timeout is None or remaining <= model_timeout
          )
          from nonoka.core.runtime import TerminalReason, Termination
          termination = Termination(
            reason=(TerminalReason.DEADLINE_EXCEEDED if deadline_limited else TerminalReason.MODEL_TIMEOUT),
            message=(
              f"Session {session.session_id} exceeded its wall-clock deadline."
              if deadline_limited else f"Model call timed out after {effective_timeout} seconds."
            ),
            dimension=("wall_timeout_seconds" if deadline_limited else "model_timeout_seconds"),
            limit=effective_timeout,
          )
          session.terminate(termination)
          raise RuntimeTerminatedError(termination) from exc
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

        session.trace.record_turn_response(turn + 1, response.model_dump())

        _logger.info(
          "llm.response",
          session_id=session.session_id,
          turn=turn + 1,
          has_tool_calls=bool(response.tool_calls),
        )

        markup_tool_call = bool(
          start_decision.disable_tools
          and response.content
          and _TOOL_CALL_MARKUP_RE.search(response.content)
        )
        if start_decision.disable_tools and (response.tool_calls or markup_tool_call):
          state = session.extension_state.setdefault("tool_free_finalization", {})
          rejected = int(state.get("rejected_tool_calls", 0))
          if rejected >= 1:
            return await self._extension_failure(
              session,
              runner,
              "The model repeatedly attempted tool calls during tool-free finalization.",
            )
          state["rejected_tool_calls"] = rejected + 1
          if session.memory is not None:
            await session.memory.add(
              "[Finalization correction] The attempted tool call was rejected and was not "
              "executed. The focused verification receipt already proves completion, even if "
              "an earlier TODO snapshot says in-progress. Tools remain disabled: do not call "
              "todowrite, bash, or any other tool and do not emit tool-call markup. Reply now "
              "with plain-prose final summary only.",
              MemoryRole.SYSTEM,
            )
          await runner.checkpoint_store.save_session(session.session_id, session.to_state())
          continue

        # --- No tool calls → final answer ---------------------------
        if not response.tool_calls:
          content = response.content or ""

          final_decision = await extension_manager.before_final_answer(LoopExtensionContext(
            session=session, runner=runner, prompt=prompt, turn=turn + 1, content=content,
          ))
          content = final_decision.replacement_content or content
          if final_decision.failure:
            return await self._extension_failure(session, runner, final_decision.failure)

          if session.memory is not None:
            await session.memory.add(content, MemoryRole.ASSISTANT)

          if final_decision.continue_loop:
            if session.memory is not None and final_decision.feedback:
              await session.memory.add(final_decision.feedback, MemoryRole.SYSTEM)
            await runner.checkpoint_store.save_session(session.session_id, session.to_state())
            continue

          contract_feedback = session.completion_feedback()
          if contract_feedback is not None:
            if session.memory is not None:
              await session.memory.add(contract_feedback, MemoryRole.SYSTEM)
            await runner.checkpoint_store.save_session(session.session_id, session.to_state())
            continue

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
        external_count = sum(
          bool(getattr(self._capability_for_call(session, tc), "external", False))
          for tc in response.tool_calls
        )
        last_tool = response.tool_calls[-1].get("function", {}).get("name", "")
        session.reserve_tool_calls(
          num_tool_calls, external_count=external_count, last_tool=last_tool,
        )
        if session.agent.max_steps is not None and session.step_count + num_tool_calls > session.agent.max_steps:
          raise MaxStepsExceeded(
            f"Max steps ({session.agent.max_steps}) exceeded for session {session.session_id}"
          )

        # Persist the assistant tool_calls message before execution so a
        # crash mid-turn leaves a checkpoint that resume() can repair.
        await runner.checkpoint_store.save_session(session.session_id, session.to_state())

        tool_results = await self._execute_tool_calls(
          session, runner, response.tool_calls, max_concurrency,
        )
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
            response_metadata = tr.get("metadata", {}) if isinstance(tr, dict) else {}
            await session.memory.add(
              obs_text,
              MemoryRole.TOOL,
              defer_budget=True,
              tool_call_id=tc_id,
              tool_name=func_name,
              context_protected=bool(response_metadata.get("context_protected")),
              skill_name=response_metadata.get("skill_name"),
              skill_directory=response_metadata.get("skill_directory"),
            )

            # Collect ToolResponse metadata to inject after all tool messages
            if isinstance(tr, dict):
              suggested = tr.get("suggested_next_step")
              if suggested:
                tool_guidance.append(f"[Tool guidance] {suggested}")

              # ``has_more`` is part of the ToolResponse protocol itself.
              # Honour an explicit terminal signal even when the tool did not
              # opt into pagination metadata; that metadata only tunes loop
              # detection for legitimate repeated page fetches.
              if tr.get("has_more") is False:
                has_more_notices.append(
                  f"[System notice] {func_name or 'the tool'} returned 'has_more': false — "
                  "there is no additional data available."
                )

        if session.memory is not None:
          await session.enforce_context_budget()
          for notice in has_more_notices + tool_guidance:
            await session.memory.add(notice, MemoryRole.SYSTEM)

        batch_decision = await extension_manager.after_tool_batch(LoopExtensionContext(
          session=session, runner=runner, prompt=prompt, turn=turn + 1,
          tool_calls=response.tool_calls, tool_results=tool_results,
        ))
        if batch_decision.failure:
          return await self._extension_failure(session, runner, batch_decision.failure)
        if batch_decision.feedback and session.memory is not None:
          await session.memory.add(batch_decision.feedback, MemoryRole.SYSTEM)

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

    except RuntimeTerminatedError as e:
      session.status = SessionStatus.CANCELLED if e.termination.reason.value == "cancelled" else SessionStatus.FAILED
      session.end_time = __import__("datetime").datetime.now()
      await runner.checkpoint_store.save_session(session.session_id, session.to_state())
      return RunResult(
        success=False,
        session=session,
        error=str(e),
        error_type=(
          "limit_exceeded"
          if e.termination.reason.value in {"turn_budget_exhausted", "tool_budget_exhausted"}
          else e.termination.reason.value
        ),
        termination=e.termination,
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
    """Resume a conversational session from checkpoint.

    A checkpoint saved mid-turn may contain an assistant tool_calls message
    without its tool results (the process crashed between the two).  Repair
    that first by re-executing the dangling calls, then continue the loop.

    Trade-off: replaying a call whose pre-crash execution did finish means
    its side effects may be applied once more; git checkpoint/rollback is
    the mitigation for workspace-mutating tools.
    """
    await self._replay_dangling_tool_calls(session, runner)
    return await self.run(session, runner, prompt="")

  async def _replay_dangling_tool_calls(self, session: Session, runner: Any) -> None:
    """Re-execute tool calls left without results by a mid-turn crash."""
    if session.memory is None:
      return

    # Find the last assistant message carrying tool_calls (same scan as
    # resume_approval) and keep only calls with no TOOL result after it.
    assistant_index: int | None = None
    pending_tool_calls: list[dict[str, Any]] = []
    for i in range(len(session.memory.entries) - 1, -1, -1):
      entry = session.memory.entries[i]
      if entry.role == MemoryRole.ASSISTANT and entry.metadata.get("tool_calls"):
        assistant_index = i
        pending_tool_calls = list(entry.metadata["tool_calls"])
        break

    if assistant_index is None or not pending_tool_calls:
      return

    answered: set[str] = set()
    for entry in session.memory.entries[assistant_index + 1:]:
      if entry.role == MemoryRole.TOOL:
        tc_id = entry.metadata.get("tool_call_id")
        if tc_id:
          answered.add(str(tc_id))

    dangling = [
      tc for tc in pending_tool_calls
      if str(tc.get("id") or tc.get("tool_call_id", "unknown")) not in answered
    ]
    if not dangling:
      return

    max_concurrency = (
      self.max_concurrency
      if self.max_concurrency is not None
      else session.agent.max_concurrency
    )
    tool_results = await self._execute_tool_calls(
      session, runner, dangling, max_concurrency,
    )

    # Mirror run(): only HALT / FAIL / cancellation terminate the loop; other
    # errors are written back as observations so the model can self-correct.
    for tr in tool_results:
      if isinstance(tr, (SafetyError, ToolFatalError, asyncio.CancelledError)):
        raise tr

    # Write results in the same format as a normal turn (see run()).
    for tc, tr in zip(dangling, tool_results):
      tc_id = tc.get("id") or tc.get("tool_call_id", "unknown")
      func_name = tc.get("function", {}).get("name", "")
      if isinstance(tr, Exception):
        obs_text = f"Error: {type(tr).__name__}: {tr}"
      else:
        obs_text = json.dumps(tr, ensure_ascii=False, default=str) if not isinstance(tr, str) else tr

      response_metadata = tr.get("metadata", {}) if isinstance(tr, dict) else {}
      await session.memory.add(
        obs_text,
        MemoryRole.TOOL,
        defer_budget=True,
        tool_call_id=tc_id,
        tool_name=func_name,
        context_protected=bool(response_metadata.get("context_protected")),
        skill_name=response_metadata.get("skill_name"),
        skill_directory=response_metadata.get("skill_directory"),
      )

    await session.enforce_context_budget()
    await runner.checkpoint_store.save_session(session.session_id, session.to_state())

  async def resume_approval(
    self,
    session: Session,
    runner: Any,
    approvals: dict[str, dict[str, Any]],
  ) -> RunResult:
    """Resume a session that is paused waiting for tool-call approvals.

    Args:
      session: The paused session (already loaded from checkpoint).
      runner: The runner that owns the checkpoint store.
      approvals: Mapping from tool_call_id to decision dict.
        Each decision must contain ``approved: bool`` and optionally
        ``modified_args: dict[str, Any]``.

    Returns:
      The final ``RunResult`` after executing approved tools and continuing
      the ReAct loop.
    """
    if session.memory is None:
      raise RuntimeError("Cannot resume approval: session has no memory.")

    # Find the last assistant message that contains pending tool_calls.
    pending_tool_calls: list[dict[str, Any]] = []
    for entry in reversed(session.memory.entries):
      if entry.role == MemoryRole.ASSISTANT and entry.metadata.get("tool_calls"):
        pending_tool_calls = list(entry.metadata["tool_calls"])
        break

    if not pending_tool_calls:
      raise RuntimeError("No pending tool calls found for approval resume.")

    # Execute each pending tool call according to the decision.
    for tc in pending_tool_calls:
      tc_id = tc.get("id") or tc.get("tool_call_id", "unknown")
      decision = approvals.get(tc_id, {"approved": False})
      approved = bool(decision.get("approved", False))
      modified_args = decision.get("modified_args")

      if not approved:
        reason = decision.get("reason", "Execution not approved.")
        result = {"error": f"Approval denied: {reason}", "approved": False}
        await session.memory.add(
          json.dumps(result, ensure_ascii=False, default=str),
          MemoryRole.TOOL,
          defer_budget=True,
          tool_call_id=tc_id,
          tool_name=tc.get("function", {}).get("name", ""),
        )
        continue

      # Apply modified args if provided.
      if modified_args is not None:
        tc = dict(tc)
        tc["function"] = dict(tc.get("function", {}))
        tc["function"]["arguments"] = modified_args

      try:
        result = await self._execute_tool_call(session, runner, tc)
      except Exception as exc:
        result = {"error": f"{type(exc).__name__}: {exc}", "approved": True}

      await session.memory.add(
        json.dumps(result, ensure_ascii=False, default=str) if not isinstance(result, str) else result,
        MemoryRole.TOOL,
        defer_budget=True,
        tool_call_id=tc_id,
        tool_name=tc.get("function", {}).get("name", ""),
      )

    await session.memory.enforce_budget()
    session.status = SessionStatus.RUNNING
    session.end_time = None
    await runner.checkpoint_store.save_session(session.session_id, session.to_state())

    # Continue the ReAct loop.  The next LLM call will see the assistant
    # tool_calls plus the newly injected tool results.
    async for _ in self.run_stream(session, runner, prompt=""):
      # We use the non-streaming wrapper because resume_approval returns a
      # single RunResult.  The stream is consumed internally.
      pass

    # Status is updated by run_stream; reconstruct a result from session.
    return RunResult(
      success=session.status == SessionStatus.COMPLETED,
      data=getattr(session, "_last_tool_result", None),
      session=session,
      error=None if session.status == SessionStatus.COMPLETED else "Approval resume did not complete",
      error_type=None if session.status == SessionStatus.COMPLETED else "approval_resume_incomplete",
    )

  async def resume_external_tools(
    self,
    session: Session,
    runner: Any,
    results: dict[str, Any],
  ) -> AsyncIterator[Any]:
    """Resume a session paused for external tool execution.

    The pending tool calls were forwarded to an external host (e.g. OpenCode)
    and their results are supplied in *results*, keyed by tool_call_id. This
    method injects the results into memory and continues the ReAct loop.

    Args:
      session: The paused session (already loaded from checkpoint).
      runner: The runner that owns the checkpoint store.
      results: Mapping from tool_call_id to the tool result returned by the
        external host.

    Yields:
      StreamEvent objects from the continued ReAct loop.
    """
    from nonoka.core.runner import StreamEvent

    if session.memory is None:
      raise RuntimeError("Cannot resume external tools: session has no memory.")
    session.check_runtime()

    # Find the last assistant message that contains pending tool_calls.
    pending_tool_calls: list[dict[str, Any]] = []
    assistant_entry_index: int | None = None
    for i in range(len(session.memory.entries) - 1, -1, -1):
      entry = session.memory.entries[i]
      if entry.role == MemoryRole.ASSISTANT and entry.metadata.get("tool_calls"):
        pending_tool_calls = list(entry.metadata["tool_calls"])
        assistant_entry_index = i
        break

    if not pending_tool_calls:
      raise RuntimeError("No pending tool calls found for external tool resume.")

    # Determine which pending tool calls already have a result in memory.
    existing_tool_ids: set[str] = set()
    existing_tool_results: dict[str, str] = {}
    if assistant_entry_index is not None:
      for entry in session.memory.entries[assistant_entry_index + 1:]:
        if entry.role == MemoryRole.TOOL:
          tc_id = entry.metadata.get("tool_call_id")
          if tc_id:
            existing_tool_ids.add(str(tc_id))
            existing_tool_results[str(tc_id)] = entry.content

    # Inject results for pending tool calls that are not already in memory.
    resumed_tool_results: list[Any] = []
    for tc in pending_tool_calls:
      tc_id = tc.get("id") or tc.get("tool_call_id", "unknown")
      tc_name = tc.get("function", {}).get("name", "")

      if tc_id in existing_tool_ids:
        resumed_tool_results.append(existing_tool_results.get(str(tc_id), ""))
        continue

      from nonoka.core.external_tool import ExternalToolReceipt, ObservationCompleteness

      raw_result = results.get(tc_id)
      capability = self._capability_for_call(session, tc)
      receipt = ExternalToolReceipt.from_value(raw_result) if raw_result is not None else None
      requires_attestation = bool(getattr(capability, "requires_workspace_attestation", False))
      if requires_attestation and (receipt is None or receipt.workspace is None):
        raise ValueError(
          f"External tool '{tc_name}' declared workspace mutation but returned no workspace attestation."
        )

      if receipt is not None:
        session.trace.record_external_receipt(
          str(tc_id), receipt, verified=receipt.workspace is not None,
        )
      result = receipt.result if receipt is not None else None
      resumed_tool_results.append(
        result
        if raw_result is not None
        else {"error": f"No result returned by external host for tool '{tc_name}'"}
      )
      if raw_result is None:
        obs_text = json.dumps(
          {"error": f"No result returned by external host for tool '{tc_name}'", "tool_call_id": tc_id},
          ensure_ascii=False,
          default=str,
        )
      elif not isinstance(result, str):
        obs_text = json.dumps(result, ensure_ascii=False, default=str)
      else:
        obs_text = result

      if receipt is not None and receipt.workspace is not None:
        violations = list(receipt.workspace.policy_violations)
        restored = set(receipt.workspace.restored_paths)
        if violations:
          usage = session.runtime_state.usage
          usage.policy_violation_count += len(violations)
          for path in violations:
            if path not in usage.policy_violations:
              usage.policy_violations.append(path)
          unrestored = [path for path in violations if path not in restored]
          if unrestored:
            from nonoka.core.errors import RuntimeTerminatedError
            from nonoka.core.runtime import TerminalReason, Termination
            termination = Termination(
              reason=TerminalReason.EXECUTION_POLICY_VIOLATION,
              message="External execution modified protected workspace inputs.",
              dimension="workspace_policy",
              diagnostics={"paths": unrestored},
            )
            session.terminate(termination)
            raise RuntimeTerminatedError(termination)
          obs_text = (
            "[Execution policy] The host restored protected workspace inputs modified by "
            f"this tool: {', '.join(violations)}. Treat these paths as immutable and use "
            "a different approach.\n\n" + obs_text
          )

      result_limit = session.runtime_state.limits.max_external_result_bytes
      original_bytes = len(obs_text.encode("utf-8"))
      was_truncated = result_limit is not None and original_bytes > result_limit

      observed_completeness: ObservationCompleteness | None = None
      observation_index: int | None = None
      if receipt is not None:
        observed_completeness = (
          ObservationCompleteness.PARTIAL if was_truncated else receipt.completeness
        )
        usage = session.runtime_state.usage
        usage.observation_count += 1
        observation_index = usage.observation_count
        if observed_completeness == ObservationCompleteness.PARTIAL:
          usage.partial_observation_count += 1
          usage.last_partial_observation_at = observation_index
          usage.latest_partial_tool = str(tc_name) or None
          usage.latest_partial_artifact_ref = receipt.artifact_ref
          obs_text = (
            "[Partial observation] This host result is not complete and cannot establish "
            "exhaustive absence or full coverage. Narrow the query or inspect a bounded "
            "artifact or region before making a completion claim.\n\n" + obs_text
          )
          fallback_guidance = self._observation_fallback_guidance(session)
          if fallback_guidance:
            obs_text = fallback_guidance + "\n\n" + obs_text
          automatic_fallback = await self._execute_partial_observation_fallback(
            session, runner, tc,
          )
          if automatic_fallback is not None:
            fallback_name, fallback_result = automatic_fallback
            fallback_text = (
              fallback_result
              if isinstance(fallback_result, str)
              else json.dumps(fallback_result, ensure_ascii=False, default=str)
            )
            obs_text = (
              f"[Automatic local observation fallback: {fallback_name}]\n"
              f"{fallback_text}\n\n" + obs_text
            )
        elif observed_completeness == ObservationCompleteness.COMPLETE:
          usage.last_complete_observation_at = observation_index
          if (
            usage.last_partial_observation_at is not None
            and observation_index > usage.last_partial_observation_at
          ):
            usage.complete_observations_after_partial += 1
        else:
          usage.unknown_observation_count += 1

      # Compact after adding framework feedback so the complete observation
      # stored in memory remains within the declared result budget.
      if was_truncated and result_limit is not None:
        marker = "\n...[external result compacted by nonoka]...\n"
        marker_bytes = len(marker.encode("utf-8"))
        side = max(1, (result_limit - marker_bytes) // 2)
        encoded = obs_text.encode("utf-8")
        obs_text = (
          encoded[:side].decode("utf-8", errors="ignore")
          + marker
          + encoded[-side:].decode("utf-8", errors="ignore")
        )

      workspace_changes = None
      workspace_changed = False
      if receipt is not None and receipt.workspace is not None:
        workspace_changes = {
          "created": list(receipt.workspace.created),
          "modified": list(receipt.workspace.modified),
          "deleted": list(receipt.workspace.deleted),
        }
        changed = sum(len(items) for items in workspace_changes.values())
        if changed:
          workspace_changed = True
          usage = session.runtime_state.usage
          usage.mutation_count += changed
          if usage.first_mutation_at is None:
            usage.first_mutation_at = __import__("datetime").datetime.now().astimezone()
          for path in [*workspace_changes["created"], *workspace_changes["modified"], *workspace_changes["deleted"]]:
            if path not in usage.changed_paths:
              usage.changed_paths.append(path)
      if receipt is not None and (
        workspace_changed or (receipt.effect is not None and receipt.effect.changed)
      ):
        usage = session.runtime_state.usage
        usage.effect_count += 1
        usage.last_effect_at_observation = observation_index
      if receipt is not None and receipt.verification is not None:
        verification = receipt.verification
        status = verification.status.value
        trustworthy = (
          verification.exit_code == 0
          and not verification.timed_out
          and not verification.truncated
          and verification.completeness == ObservationCompleteness.COMPLETE
          and (
            verification.kind.value != "test"
            or (
              verification.collected_tests is not None
              and verification.collected_tests > 0
            )
          )
        )
        if status == "passed" and not trustworthy:
          status = "unavailable"
        elif verification.timed_out or verification.exit_code is None:
          status = "unavailable"
        elif verification.exit_code != 0:
          status = "failed"

        verification_data = {
          "status": status,
          "level": verification.level.value,
          "kind": verification.kind.value,
          "command": verification.command,
          "cwd": verification.cwd,
          "exit_code": verification.exit_code,
          "timed_out": verification.timed_out,
          "timeout_seconds": verification.timeout_seconds,
          "truncated": verification.truncated,
          "completeness": verification.completeness.value,
          "collected_tests": verification.collected_tests,
          "summary": verification.summary,
          "failure_summary": verification.failure_summary,
          "artifact_ref": verification.artifact_ref,
        }
        usage = session.runtime_state.usage
        usage.verification_count += 1
        usage.verification_status = status
        usage.latest_verification = verification_data
        usage.latest_verification_at_observation = observation_index
        if verification.level.value == "focused":
          usage.focused_verification_status = status
          if status == "passed":
            usage.last_passed_focused_at_observation = observation_index
          elif status == "failed":
            usage.last_failed_focused_at_observation = observation_index
          elif status == "unavailable":
            usage.last_unavailable_focused_at_observation = observation_index
        else:
          usage.full_verification_status = status
        session.trace.record_verification(source="external_host", **verification_data)
        label = f"{verification.level.value} {verification.kind.value} verification"
        obs_text = f"[Verification: {label} {status}]\n{obs_text}"
      if receipt is not None and receipt.exit_code == 0:
        raw_arguments = tc.get("function", {}).get("arguments", {})
        if isinstance(raw_arguments, str):
          try:
            raw_arguments = json.loads(raw_arguments)
          except json.JSONDecodeError:
            raw_arguments = {}
        if isinstance(raw_arguments, dict):
          command = raw_arguments.get("command")
          if isinstance(command, str):
            usage = session.runtime_state.usage
            if command not in usage.verified_commands:
              usage.verified_commands.append(command)
            usage.successful_command_count += 1
            usage.last_successful_command = command
            usage.last_successful_command_at_observation = observation_index

      await session.memory.add(
        obs_text,
        MemoryRole.TOOL,
        defer_budget=True,
        tool_call_id=tc_id,
        tool_name=tc_name,
        exit_code=receipt.exit_code if receipt is not None else None,
        artifact_ref=receipt.artifact_ref if receipt is not None else None,
        output_kind=receipt.output_kind if receipt is not None else None,
        original_bytes=(receipt.original_bytes if receipt and receipt.original_bytes else original_bytes),
        truncated=(receipt.truncated if receipt is not None else False) or was_truncated,
        completeness=observed_completeness.value if observed_completeness is not None else None,
        workspace_changes=workspace_changes,
      )

    await session.enforce_context_budget()

    # External execution is one logical tool batch. Run extension and loop
    # decisions only after every TOOL receipt has been inserted, preserving
    # the provider-required ASSISTANT(tool_calls) -> TOOL* adjacency.
    extension_manager = LoopExtensionManager([
      *list(getattr(session.agent, "extensions", [])), *self.extensions,
    ])
    batch_decision = await extension_manager.after_tool_batch(LoopExtensionContext(
      session=session,
      runner=runner,
      prompt="",
      turn=session.turn_count,
      tool_calls=pending_tool_calls,
      tool_results=resumed_tool_results,
    ))
    if batch_decision.failure:
      session.status = SessionStatus.FAILED
      session.end_time = __import__("datetime").datetime.now()
      await runner.checkpoint_store.save_session(session.session_id, session.to_state())
      yield StreamEvent(type="error", data={
        "success": False,
        "error": batch_decision.failure,
        "error_type": "extension_rejected",
      })
      return
    if batch_decision.feedback:
      await session.memory.add(batch_decision.feedback, MemoryRole.SYSTEM)

    if await self._detect_and_break_loops(
      session, pending_tool_calls, resumed_tool_results,
    ):
      session.status = SessionStatus.FAILED
      session.end_time = __import__("datetime").datetime.now()
      await runner.checkpoint_store.save_session(session.session_id, session.to_state())
      yield StreamEvent(type="error", data={
        "success": False,
        "error": "Agent loop detected during external tool execution.",
        "error_type": "loop_detected",
      })
      return

    session.status = SessionStatus.RUNNING
    session.end_time = None
    await runner.checkpoint_store.save_session(session.session_id, session.to_state())

    # Continue the ReAct loop.
    async for event in self.run_stream(session, runner, prompt=""):
      yield event

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
    extension_manager = LoopExtensionManager([
      *list(getattr(session.agent, "extensions", [])), *self.extensions,
    ])
    try:
      while True:
        turn = session.begin_model_turn() - 1
        await session.enforce_context_budget()

        start_decision = await extension_manager.before_turn(LoopExtensionContext(
          session=session, runner=runner, prompt=prompt, turn=turn + 1,
        ))
        if start_decision.failure:
          session.status = SessionStatus.FAILED
          session.end_time = __import__("datetime").datetime.now()
          await runner.checkpoint_store.save_session(session.session_id, session.to_state())
          yield StreamEvent(type="error", data={
            "success": False, "error": start_decision.failure, "error_type": "extension_rejected",
          })
          return
        if start_decision.feedback and session.memory is not None:
          await session.memory.add(start_decision.feedback, MemoryRole.SYSTEM)

        # Build messages from memory (or directly from prompt if no memory)
        if session.memory is not None:
          messages = (
            self._build_tool_free_finalization_messages(session, start_decision.feedback)
            if start_decision.disable_tools
            else self._build_messages(session)
          )
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
        if start_decision.disable_tools:
          available_tools = []
        tools = [t.to_json_schema() for t in available_tools] if available_tools else None

        # --- Streaming LLM call --------------------------------------
        accumulated_content = ""
        accumulated_tool_calls: dict[int, dict[str, Any]] = {}
        reported_tool_argument_chars: dict[int, int] = {}
        streamed_usage: dict[str, Any] = {}

        # Hook: llm request (streaming)
        hook_ctx = HookContext(session=session, runner=runner)
        session.trace.record_turn_request(
          turn + 1,
          [message.model_dump(exclude_none=True) for message in messages],
          tools=tools, temperature=session.agent.temperature,
          max_tokens=session.agent.max_tokens,
        )
        await runner.hooks.emit_llm_request(hook_ctx, messages, tools)

        try:
          stream = runner.llm.chat_stream(
            messages=messages, tools=tools or None,
            temperature=session.agent.temperature, max_tokens=session.agent.max_tokens,
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

        model_timeout = session.runtime_state.limits.model_timeout_seconds
        remaining = session.runtime_state.remaining_seconds()
        effective_timeout = min(
          value for value in (model_timeout, remaining) if value is not None
        ) if model_timeout is not None or remaining is not None else None
        try:
          timeout_scope = anyio.fail_after(effective_timeout) if effective_timeout is not None else None
          if timeout_scope is None:
            async for chunk in stream:
              usage = getattr(chunk, "usage", None)
              if usage:
                streamed_usage.update(usage)
              if chunk.content_delta:
                accumulated_content += chunk.content_delta
                if not start_decision.disable_tools:
                  yield StreamEvent(type="content_delta", data={"content": chunk.content_delta})
              if chunk.tool_call_deltas:
                self._accumulate_tool_deltas(accumulated_tool_calls, chunk.tool_call_deltas)
                if not start_decision.disable_tools:
                  for progress in self._collect_tool_call_progress(
                    accumulated_tool_calls, reported_tool_argument_chars,
                  ):
                    yield StreamEvent(type="tool_call_progress", data=progress)
              if chunk.finish_reason:
                break
          else:
            with timeout_scope:
              async for chunk in stream:
                usage = getattr(chunk, "usage", None)
                if usage:
                  streamed_usage.update(usage)
                if chunk.content_delta:
                  accumulated_content += chunk.content_delta
                  if not start_decision.disable_tools:
                    yield StreamEvent(type="content_delta", data={"content": chunk.content_delta})
                if chunk.tool_call_deltas:
                  self._accumulate_tool_deltas(accumulated_tool_calls, chunk.tool_call_deltas)
                  if not start_decision.disable_tools:
                    for progress in self._collect_tool_call_progress(
                      accumulated_tool_calls, reported_tool_argument_chars,
                    ):
                      yield StreamEvent(type="tool_call_progress", data=progress)
                if chunk.finish_reason:
                  break
        except TimeoutError as exc:
          deadline_limited = remaining is not None and (
            model_timeout is None or remaining <= model_timeout
          )
          from nonoka.core.runtime import TerminalReason, Termination
          termination = Termination(
            reason=(TerminalReason.DEADLINE_EXCEEDED if deadline_limited else TerminalReason.MODEL_TIMEOUT),
            message=(
              f"Session {session.session_id} exceeded its wall-clock deadline."
              if deadline_limited else f"Model call timed out after {effective_timeout} seconds."
            ),
            dimension=("wall_timeout_seconds" if deadline_limited else "model_timeout_seconds"),
            limit=effective_timeout,
          )
          session.terminate(termination)
          raise RuntimeTerminatedError(termination) from exc

        tool_calls = self._finalize_tool_calls(accumulated_tool_calls)
        response = LLMResponse(
          content=accumulated_content or None,
          tool_calls=tool_calls or None,
          usage=streamed_usage,
        )
        session.trace.record_turn_response(turn + 1, response.model_dump())

        # Hook: llm response (streaming)
        await runner.hooks.emit_llm_response(hook_ctx, response)
        if getattr(type(runner), "record_llm_usage", None) is not None:
          await runner.record_llm_usage(session, response.usage)

        _logger.info(
          "llm.stream_response",
          session_id=session.session_id,
          turn=turn + 1,
          has_tool_calls=bool(tool_calls),
        )

        markup_tool_call = bool(
          start_decision.disable_tools
          and response.content
          and _TOOL_CALL_MARKUP_RE.search(response.content)
        )
        if start_decision.disable_tools and (response.tool_calls or markup_tool_call):
          state = session.extension_state.setdefault("tool_free_finalization", {})
          rejected = int(state.get("rejected_tool_calls", 0))
          if rejected >= 1:
            session.status = SessionStatus.FAILED
            session.end_time = __import__("datetime").datetime.now()
            await runner.checkpoint_store.save_session(session.session_id, session.to_state())
            yield StreamEvent(type="error", data={
              "success": False,
              "error": "The model repeatedly attempted tool calls during tool-free finalization.",
              "error_type": "extension_rejected",
            })
            return
          state["rejected_tool_calls"] = rejected + 1
          if session.memory is not None:
            await session.memory.add(
              "[Finalization correction] The attempted tool call was rejected and was not "
              "executed. The focused verification receipt already proves completion, even if "
              "an earlier TODO snapshot says in-progress. Tools remain disabled: do not call "
              "todowrite, bash, or any other tool and do not emit tool-call markup. Reply now "
              "with plain-prose final summary only.",
              MemoryRole.SYSTEM,
            )
          await runner.checkpoint_store.save_session(session.session_id, session.to_state())
          continue

        # --- No tool calls → final answer ---------------------------
        if not response.tool_calls:
          content = response.content or ""

          final_decision = await extension_manager.before_final_answer(LoopExtensionContext(
            session=session, runner=runner, prompt=prompt, turn=turn + 1, content=content,
          ))
          content = final_decision.replacement_content or content
          if final_decision.failure:
            session.status = SessionStatus.FAILED
            session.end_time = __import__("datetime").datetime.now()
            await runner.checkpoint_store.save_session(session.session_id, session.to_state())
            yield StreamEvent(type="error", data={
              "success": False, "error": final_decision.failure, "error_type": "extension_rejected",
            })
            return

          if session.memory is not None:
            await session.memory.add(content, MemoryRole.ASSISTANT)

          if final_decision.continue_loop:
            if session.memory is not None and final_decision.feedback:
              await session.memory.add(final_decision.feedback, MemoryRole.SYSTEM)
            await runner.checkpoint_store.save_session(session.session_id, session.to_state())
            continue

          contract_feedback = session.completion_feedback()
          if contract_feedback is not None:
            if session.memory is not None:
              await session.memory.add(contract_feedback, MemoryRole.SYSTEM)
            await runner.checkpoint_store.save_session(session.session_id, session.to_state())
            continue

          parsed_data: Any = content
          if session.agent.result_type is not None:
            parsed_data = await self._parse_result_type(session, runner, content, turn)
            if parsed_data is None:
              continue

          session.status = SessionStatus.COMPLETED
          session.end_time = __import__("datetime").datetime.now()
          await runner.checkpoint_store.save_session(session.session_id, session.to_state())
          final_data = self._extract_result_data(session, parsed_data)
          duration = (
            (session.end_time - session.start_time).total_seconds()
            if session.start_time is not None
            else None
          )
          if start_decision.disable_tools and content:
            yield StreamEvent(type="content_delta", data={"content": content})
          yield StreamEvent(
            type="final",
            data={
              "success": True,
              "data": final_data,
              "turn_count": session.turn_count,
              "tool_call_count": session.step_count,
              "duration_seconds": duration,
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

        # Build a metadata map so hosts can route external tool calls without
        # having to re-derive server/skill names from the prefixed tool name.
        tool_metadata: dict[str, dict[str, Any]] = {}
        for t in session.agent.tools:
          meta = getattr(t, "metadata", None)
          if meta:
            tool_metadata[t.name] = dict(meta)

        enriched_tool_calls = []
        for tc in response.tool_calls:
          tc_name = tc.get("function", {}).get("name", "")
          capability = self._capability_for_call(session, tc)
          enriched = dict(tc)
          enriched["metadata"] = tool_metadata.get(tc_name) or {}
          # A capability may run inside the framework while its session is
          # hosted by an external UI.  Such a capability can opt out of host
          # lifecycle events without changing its execution semantics.  This
          # keeps private, local observations out of host tool registries.
          enriched["host_visible"] = bool(getattr(capability, "host_visible", True))
          enriched_tool_calls.append(enriched)

        host_visible_calls = [
          call for call in enriched_tool_calls if call.get("host_visible", True)
        ]
        if host_visible_calls:
          yield StreamEvent(
            type="tool_call_start",
            data={"tool_calls": host_visible_calls},
          )

        num_tool_calls = len(response.tool_calls)
        external_count = sum(
          bool(getattr(self._capability_for_call(session, tc), "external", False))
          for tc in response.tool_calls
        )
        last_tool = response.tool_calls[-1].get("function", {}).get("name", "")
        session.reserve_tool_calls(
          num_tool_calls, external_count=external_count, last_tool=last_tool,
        )
        if session.agent.max_steps is not None and session.step_count + num_tool_calls > session.agent.max_steps:
          raise MaxStepsExceeded(
            f"Max steps ({session.agent.max_steps}) exceeded for session {session.session_id}"
          )

        tool_results = await self._execute_tool_calls(
          session, runner, response.tool_calls, max_concurrency,
        )

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

        # External tools: execute any local calls from the same model response
        # before pausing for host work.  This is essential for mixed tool
        # batches: a local capability must not be silently delegated merely
        # because a sibling call is external.  ``resume_external_tools``
        # already recognises tool results persisted after the assistant call,
        # so it injects only the genuinely pending external receipts later.
        if any(isinstance(item, ExternalToolExecutionRequiredError) for item in tool_results):
          last_local_result: Any = None
          for tc, item in zip(response.tool_calls, tool_results):
            if isinstance(item, ExternalToolExecutionRequiredError):
              continue
            tc_id = tc.get("id") or tc.get("tool_call_id", "unknown")
            tc_name = tc.get("function", {}).get("name", "")
            capability = self._capability_for_call(session, tc)
            host_visible = bool(getattr(capability, "host_visible", True))
            if isinstance(item, Exception):
              obs_text = f"Error: {type(item).__name__}: {item}"
              yield StreamEvent(
                type="tool_call_result",
                data={
                  "tool_call_id": tc_id,
                  "name": tc_name,
                  "result_preview": obs_text[:500],
                  "is_error": True,
                  "host_visible": host_visible,
                },
              )
            else:
              last_local_result = item
              obs_text = (
                json.dumps(item, ensure_ascii=False, default=str)
                if not isinstance(item, str) else item
              )
              yield StreamEvent(
                type="tool_call_result",
                data={
                  "tool_call_id": tc_id,
                  "name": tc_name,
                  "result_preview": str(item)[:500],
                  "result": item,
                  "is_error": False,
                  "host_visible": host_visible,
                },
              )
            if session.memory is not None:
              response_metadata = item.get("metadata", {}) if isinstance(item, dict) else {}
              await session.memory.add(
                obs_text,
                MemoryRole.TOOL,
                defer_budget=True,
                tool_call_id=tc_id,
                tool_name=tc_name,
                context_protected=bool(response_metadata.get("context_protected")),
                skill_name=response_metadata.get("skill_name"),
                skill_directory=response_metadata.get("skill_directory"),
              )
          session._last_tool_result = last_local_result  # type: ignore[attr-defined]

          if session.memory is not None:
            await session.enforce_context_budget()
          session.status = SessionStatus.PAUSED
          session.end_time = __import__("datetime").datetime.now()
          await runner.checkpoint_store.save_session(session.session_id, session.to_state())

          yield StreamEvent(
            type="final",
            data={
              "success": False,
              "requires_external_execution": True,
              "error": "Pending external tool execution",
              "error_type": "external_tool_execution_required",
            },
          )
          return

        # HITL: if any tool call requires approval, pause the turn.
        # The assistant message with tool_calls is already saved in memory,
        # so we can resume later with the user's decisions.
        if any(isinstance(item, ApprovalRequiredError) for item in tool_results):
          session.status = SessionStatus.PAUSED
          session.end_time = __import__("datetime").datetime.now()
          await runner.checkpoint_store.save_session(session.session_id, session.to_state())

          for tc in response.tool_calls:
            func = tc.get("function", {})
            tc_id = tc.get("id") or tc.get("tool_call_id", "unknown")
            tc_name = func.get("name", "")
            tc_args = func.get("arguments", "{}")
            if isinstance(tc_args, str):
              try:
                tc_args = json.loads(tc_args)
              except json.JSONDecodeError:
                tc_args = {}
            yield StreamEvent(
              type="approval_request",
              data={
                "id": f"approval_{tc_id}_{__import__('uuid').uuid4().hex[:8]}",
                "tool_call_id": tc_id,
                "tool_name": tc_name,
                "args": tc_args,
              },
            )

          yield StreamEvent(
            type="final",
            data={
              "success": False,
              "requires_approval": True,
              "error": "Pending human approval",
              "error_type": "approval_required",
            },
          )
          return

        last_tool_result: Any = None
        for tr in reversed(tool_results):
          if not isinstance(tr, Exception):
            last_tool_result = tr
            break
        session._last_tool_result = last_tool_result  # type: ignore[attr-defined]

        for tc, item in zip(response.tool_calls, tool_results):
          tc_id = tc.get("id") or tc.get("tool_call_id", "unknown")
          tc_name = tc.get("function", {}).get("name", "")

          if isinstance(item, Exception):
            # Non-fatal exception (e.g. ValueError from missing tool) —
            # stream as an error result so the LLM can self-correct.
            obs_text = f"Error: {type(item).__name__}: {item}"
            yield StreamEvent(
              type="tool_call_result",
              data={
                "tool_call_id": tc_id,
                "name": tc_name,
                "result_preview": obs_text[:500],
                "is_error": True,
                "host_visible": bool(getattr(self._capability_for_call(session, tc), "host_visible", True)),
              },
            )
            if session.memory is not None:
              await session.memory.add(
                obs_text,
                MemoryRole.TOOL,
                defer_budget=True,
                tool_call_id=tc_id,
                tool_name=tc_name,
              )
            continue

          tr = item
          obs_text = json.dumps(tr, ensure_ascii=False, default=str) if not isinstance(tr, str) else tr

          yield StreamEvent(
            type="tool_call_result",
            data={
              "tool_call_id": tc_id,
              "name": tc_name,
              "result_preview": str(tr)[:500],
              "result": tr,
              "is_error": False,
              "host_visible": bool(getattr(self._capability_for_call(session, tc), "host_visible", True)),
            },
          )

          if session.memory is not None:
            await session.memory.add(
              obs_text,
              MemoryRole.TOOL,
              defer_budget=True,
              tool_call_id=tc_id,
              tool_name=tc_name,
            )

        if session.memory is not None:
          await session.enforce_context_budget()

        # Inject ToolResponse metadata SYSTEM messages *after* all TOOL
        # entries so that ASSISTANT+tool_calls stay contiguous with their
        # corresponding TOOL responses.
        if session.memory is not None:
          stream_guidance: list[str] = []
          stream_notices: list[str] = []
          for tc, item in zip(response.tool_calls, tool_results):
            if isinstance(item, Exception):
              continue
            tr = item
            if isinstance(tr, dict):
              suggested = tr.get("suggested_next_step")
              if suggested:
                stream_guidance.append(f"[Tool guidance] {suggested}")
              func_name = tc.get("function", {}).get("name", "the tool")
              if tr.get("has_more") is False:
                stream_notices.append(
                  f"[System notice] {func_name} returned 'has_more': false — "
                  "there is no additional data available."
                )
          for notice in stream_notices + stream_guidance:
            await session.memory.add(notice, MemoryRole.SYSTEM)

        batch_decision = await extension_manager.after_tool_batch(LoopExtensionContext(
          session=session, runner=runner, prompt=prompt, turn=turn + 1,
          tool_calls=response.tool_calls, tool_results=tool_results,
        ))
        if batch_decision.failure:
          session.status = SessionStatus.FAILED
          session.end_time = __import__("datetime").datetime.now()
          await runner.checkpoint_store.save_session(session.session_id, session.to_state())
          yield StreamEvent(type="error", data={
            "success": False, "error": batch_decision.failure, "error_type": "extension_rejected",
          })
          return
        if batch_decision.feedback and session.memory is not None:
          await session.memory.add(batch_decision.feedback, MemoryRole.SYSTEM)

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
              "error": "Agent loop detected: tool was called repeatedly without meaningful progress.",
              "error_type": "loop_detected",
            },
          )
          return

        await runner.checkpoint_store.save_session(session.session_id, session.to_state())

    except RuntimeTerminatedError as e:
      session.status = SessionStatus.CANCELLED if e.termination.reason.value == "cancelled" else SessionStatus.FAILED
      session.end_time = __import__("datetime").datetime.now()
      await runner.checkpoint_store.save_session(session.session_id, session.to_state())
      yield StreamEvent(
        type="error",
        data={
          "success": False,
          "error": str(e),
          "error_type": (
            "limit_exceeded"
            if e.termination.reason.value in {"turn_budget_exhausted", "tool_budget_exhausted"}
            else e.termination.reason.value
          ),
          "termination": e.termination.model_dump(mode="json"),
        },
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
  def _collect_tool_call_progress(
    accumulator: dict[int, dict[str, Any]],
    reported_argument_chars: dict[int, int],
  ) -> list[dict[str, Any]]:
    """Return bounded, content-free progress for streaming tool arguments."""
    progress: list[dict[str, Any]] = []
    for idx in sorted(accumulator):
      function = accumulator[idx].get("function", {})
      argument_chars = len(function.get("arguments", ""))
      if argument_chars <= 0:
        continue

      last_reported = reported_argument_chars.get(idx)
      if (
        last_reported is not None
        and argument_chars - last_reported < _TOOL_CALL_PROGRESS_INTERVAL_CHARS
      ):
        continue

      reported_argument_chars[idx] = argument_chars
      progress.append({
        "tool_call_index": idx,
        "tool_name": function.get("name", ""),
        "argument_chars": argument_chars,
      })
    return progress

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

  def _build_tool_free_finalization_messages(
    self,
    session: Session,
    feedback: str | None,
  ) -> list[LLMMessage]:
    """Build a compact final-answer context without executable history.

    A tool-free finalization turn must not replay either the normal system
    prompt or assistant/tool-call history: both can instruct or prime the model
    to continue an action whose authority has already closed. Preserve the
    original user goal and the host-attested latest verification receipt so the
    summary remains grounded without exposing another executable trajectory.
    """
    corrections: list[str] = []
    user_requests: list[str] = []
    if session.memory is not None:
      corrections.extend(
        entry.content
        for entry in session.memory.entries
        if entry.role == MemoryRole.SYSTEM
        and entry.content.startswith("[Finalization correction]")
      )
      user_requests.extend(
        entry.content for entry in session.memory.entries if entry.role == MemoryRole.USER
      )
    instruction = "\n\n".join(
      part for part in [feedback, *corrections] if part
    ) or (
      "[Finalization turn] Tools are disabled. Return a plain-prose final answer "
      "grounded in the user request and tool evidence."
    )
    runtime_state = getattr(session, "runtime_state", None)
    usage = getattr(runtime_state, "usage", None)
    verification = getattr(usage, "latest_verification", None)
    evidence = (
      json.dumps(verification, ensure_ascii=False, sort_keys=True)
      if isinstance(verification, dict)
      else "The configured completion contract is satisfied."
    )
    request = "\n\n".join(user_requests) or "Summarize the completed task."
    return [
      LLMMessage(role=LLMMessageRole.SYSTEM, content=instruction),
      LLMMessage(
        role=LLMMessageRole.USER,
        content=f"{request}\n\n[Host-attested completion evidence]\n{evidence}",
      ),
    ]

  def _capability_for_call(self, session: Session, tool_call: dict[str, Any]) -> Any | None:
    name = tool_call.get("function", {}).get("name", "")
    return next((tool for tool in session.agent.tools if tool.name == name), None)

  @staticmethod
  def _observation_fallback_guidance(session: Session) -> str:
    """Describe registered local fallbacks for a partial host observation."""
    available: list[str] = []
    for tool in session.agent.tools:
      metadata = getattr(tool, "metadata", None)
      if not isinstance(metadata, dict) or metadata.get("kind") != "observation_fallback":
        continue
      description = str(getattr(tool, "description", "")).strip()
      available.append(f"{tool.name}: {description}" if description else str(tool.name))
    if not available:
      return ""
    return (
      "[Local observation fallback] The preceding host observation was partial. "
      "Before relying on it or retrying the same broad host query, use one of these "
      "local capabilities for bounded evidence: " + "; ".join(available)
    )

  async def _execute_partial_observation_fallback(
    self,
    session: Session,
    runner: Any,
    source_call: dict[str, Any],
  ) -> tuple[str, Any] | None:
    """Run one declared read-only fallback for a partial external receipt.

    A fallback is opt-in capability metadata, not a host-name heuristic. Its
    ``fallback`` declaration maps source-call argument names to the fallback's
    input names and supplies any static defaults. This lets a framework host
    provide bounded local evidence without knowing which external search tool
    produced the incomplete observation.
    """
    source_args = source_call.get("function", {}).get("arguments", {})
    if isinstance(source_args, str):
      try:
        source_args = json.loads(source_args)
      except json.JSONDecodeError:
        return None
    if not isinstance(source_args, dict):
      return None

    for capability in session.agent.tools:
      metadata = getattr(capability, "metadata", None)
      if not isinstance(metadata, dict) or metadata.get("kind") != "observation_fallback":
        continue
      declaration = metadata.get("fallback")
      if not isinstance(declaration, dict) or not declaration.get("on_partial_external"):
        continue
      if not execution_for(capability).read_only:
        continue
      argument_map = declaration.get("argument_map", {})
      defaults = declaration.get("defaults", {})
      if not isinstance(argument_map, dict) or not isinstance(defaults, dict):
        continue
      arguments = dict(defaults)
      if any(not isinstance(target, str) or source not in source_args for target, source in argument_map.items()):
        continue
      arguments.update({target: source_args[source] for target, source in argument_map.items()})

      synthetic_call = {
        "id": f"fallback_{source_call.get('id') or source_call.get('tool_call_id') or 'unknown'}",
        "type": "function",
        "function": {"name": capability.name, "arguments": json.dumps(arguments)},
      }
      try:
        session.reserve_tool_calls(1, last_tool=capability.name)
        result = await self._execute_tool_call(session, runner, synthetic_call)
      except Exception as exc:
        return capability.name, {"error": f"{type(exc).__name__}: {exc}"}
      return capability.name, result
    return None

  def _is_pagination_tool(self, session: Session, tool_name: str) -> bool:
    capability = next((tool for tool in session.agent.tools if tool.name == tool_name), None)
    return execution_for(capability).pagination

  async def _execute_tool_calls(
    self,
    session: Session,
    runner: Any,
    tool_calls: list[dict[str, Any]],
    max_concurrency: int,
  ) -> list[Any]:
    coordinator = ToolExecutionCoordinator(max_concurrency)
    return await coordinator.execute(
      tool_calls,
      lambda call: self._capability_for_call(session, call),
      lambda call: self._execute_tool_call(session, runner, call),
    )

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

    trace_index = session.trace.record_tool_start(
      str(tool_call.get("id") or tool_call.get("tool_call_id") or "unknown"),
      name, arguments, execution_for(capability),
    )

    ctx = RunContext(session)
    session.step_count += 1

    # Emit event
    ctx.emit(AgentEvent(
      type=EventType.TOOL_CALLED,
      session_id=session.session_id,
      data={"tool": name, "arguments": arguments},
    ))

    # Hook: tool start (notification)
    hook_ctx = HookContext(session=session, runner=runner)
    await runner.hooks.emit_tool_start(hook_ctx, name, arguments)

    # Hook: tool start intercept (can modify arguments)
    arguments = await runner.hooks.emit_tool_start_intercept(hook_ctx, name, arguments)

    # External tools are executed by the host (e.g. OpenCode), not by nonoka.
    if getattr(capability, "external", False):
      exc = ExternalToolExecutionRequiredError(
        tool_call_id=tool_call.get("id") or tool_call.get("tool_call_id", "unknown"),
        tool_name=name,
        arguments=arguments,
      )
      session.trace.record_tool_end(trace_index, error=exc)
      raise exc

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
    session.trace.record_tool_end(trace_index, result=result, error=error)

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

  async def _extension_failure(self, session: Session, runner: Any, error: str) -> RunResult:
    """Finish a run rejected by a bounded loop extension."""
    session.status = SessionStatus.FAILED
    session.end_time = __import__("datetime").datetime.now()
    await runner.checkpoint_store.save_session(session.session_id, session.to_state())
    return RunResult(
      success=False, session=session, error=error, error_type="extension_rejected",
    )

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
    tool_name = history[-1][0] if history else "unknown"
    capability = next((tool for tool in session.agent.tools if tool.name == tool_name), None)
    execution = execution_for(capability)

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

    if execution.pagination and _has_more_true(len(result_history) - 1):
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
        if execution.pagination and _has_more_true(prev_idx):
          consecutive_count += 0.5
        else:
          consecutive_count += 1
      else:
        break

    # Accelerate detection when has_more=false but tool is still called
    h1_threshold = self.max_repeated_tool_calls
    if history and _has_more_false(len(result_history) - 1):
      h1_threshold = max(2, h1_threshold - 1)

    # One model response may intentionally batch the same read-only tool over
    # different files. That is parallel exploration, not a repeated
    # action-observation cycle. Exact duplicates remain covered by the
    # signature heuristic below.
    h1_triggered = (
      len(current_sigs) == 1
      and not execution.is_stateful
      and consecutive_count >= h1_threshold
    )

    # -- Heuristic 2: repeated (name, args) pair --------------------------
    recent = history[-10:]
    repeat_counts: dict[tuple[str, str], int] = {}
    for sig in recent:
      repeat_counts[sig] = repeat_counts.get(sig, 0) + 1
    max_repeat = max(repeat_counts.values()) if repeat_counts else 0
    h2_triggered = max_repeat >= self.loop_similarity_threshold
    if execution.is_stateful:
      last_sig = history[-1] if history else ("", "")
      last_result = result_history[-1] if result_history else None
      same_outcome = sum(
        1 for index, sig in enumerate(history)
        if sig == last_sig
        and index < len(result_history)
        and _results_similar(result_history[index], last_result)
        and not (isinstance(result_history[index], dict) and result_history[index].get("progress") is True)
      )
      h2_triggered = same_outcome >= self.loop_similarity_threshold

    # -- Heuristic 3: short-cycle detection (A→B→A→B, A→B→C→A→B→C) -----
    h3_triggered = False
    if not execution.is_stateful and len(history) >= 4:
      names = [s[0] for s in history[-4:]]
      if names[0] == names[2] and names[1] == names[3] and names[0] != names[1]:
        h3_triggered = True
    if not execution.is_stateful and len(history) >= 6 and not h3_triggered:
      names = [s[0] for s in history[-6:]]
      if names[:3] == names[3:]:
        h3_triggered = True

    # -- Heuristic 4: result similarity -----------------------------------
    h4_triggered = False
    if not execution.is_stateful and len(result_history) >= 3 and len(history) >= 3:
      last_tool = history[-1][0]
      # Only consider the most recent *consecutive* calls to the same tool.
      # A different tool in between resets the window, which matches user
      # expectations (e.g. search → edit → search is not a loop).
      consecutive_same_tool_results: list[Any] = []
      for i in range(len(history) - 1, -1, -1):
        if history[i][0] == last_tool:
          consecutive_same_tool_results.insert(0, result_history[i])
        else:
          break

      if len(consecutive_same_tool_results) >= 3:
        window = consecutive_same_tool_results[-3:]
        # Pagination in progress — don't penalise similar-looking pages.
        if not any(
          isinstance(r, dict) and r.get("has_more") is True for r in window
        ):
          if (
            _results_similar(window[0], window[1])
            and _results_similar(window[1], window[2])
          ):
            h4_triggered = True

    loop_detected = h1_triggered or h2_triggered or h3_triggered or h4_triggered

    if not loop_detected:
      return False

    # -- Escalation -------------------------------------------------------
    session._loop_trigger_count += 1  # type: ignore[attr-defined]
    trigger_count: int = session._loop_trigger_count  # type: ignore[attr-defined]

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

        pending_steps = [plan.get_step(sid) for sid in pending_ids if plan.get_step(sid)]
        coordinator = ToolExecutionCoordinator(self.max_concurrency)
        results = await coordinator.execute(
          [{"step": step} for step in pending_steps],
          lambda item: next((tool for tool in session.agent.tools if tool.name == item["step"].tool), None),
          lambda item: self._execute_step(item["step"], session, runner),
        )

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

    # Hook: plan step start (notification)
    hook_ctx = HookContext(session=session, runner=runner)
    await runner.hooks.emit_plan_step_start(hook_ctx, step.id, step.tool, resolved_args)

    # Hook: tool start intercept (can modify arguments, shared with ReAct)
    resolved_args = await runner.hooks.emit_tool_start_intercept(hook_ctx, step.tool, resolved_args)

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
    details: dict[str, Any] | None = None,
  ):
    self.passed = passed
    self.feedback = feedback
    self.score = score
    self.details = details or {}


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
      if hasattr(session, "trace"):
        session.trace.record_verification(
          iteration=iteration,
          passed=eval_result.passed,
          score=eval_result.score,
          feedback=eval_result.feedback,
        )

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
