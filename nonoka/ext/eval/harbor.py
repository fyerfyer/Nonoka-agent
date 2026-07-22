"""Harbor / Terminal-Bench 2 external-agent adapter.

Harbor owns the task container and official verifier.  Nonoka owns only the
ReAct decision loop and invokes the supplied ``BaseEnvironment`` for every
terminal action.  Importing this module without Harbor installed remains safe
so the core package has no optional benchmark dependency.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from nonoka import Agent, Runner, ToolResponse, tool
from nonoka.core.execution import ToolExecution
from nonoka.ext.coding import WorkspaceProgressExtension


_DEFAULT_TERMINAL_OUTPUT_CHARS = 12_000
_TERMINAL_SYSTEM_PROMPT = (
  "You are solving a Terminal-Bench task in a Linux environment. Inspect before editing, "
  "execute commands through the terminal tool, verify task-local results, and stop when complete. "
  "Treat user-supplied literal replacements, paths, formats, and filenames as acceptance criteria: "
  "preserve their spelling byte-for-byte instead of inventing a generic substitute. Before stopping, "
  "verify both that forbidden values are absent and that every requested replacement is present. "
  "Treat every positive search result as a required edit-and-verify checklist item; do not exclude a "
  "discovered file merely because it is large, generated, or outside the first source directory."
)

try:  # pragma: no cover - exercised in the dedicated Harbor environment
  from harbor.agents.base import BaseAgent
except ImportError as exc:  # pragma: no cover
  _HARBOR_IMPORT_ERROR: ImportError | None = exc
  BaseAgent = object  # type: ignore[misc,assignment]
else:  # pragma: no cover
  _HARBOR_IMPORT_ERROR = None


def trace_metrics(trace: dict[str, Any] | None) -> dict[str, int]:
  metrics = {"llm_calls": 0, "tool_calls": 0, "input_tokens": 0, "output_tokens": 0}
  for turn in (trace or {}).get("turns", []):
    response = turn.get("response", {})
    usage = response.get("usage", {}) if isinstance(response, dict) else {}
    metrics["llm_calls"] += 1
    metrics["input_tokens"] += int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    metrics["output_tokens"] += int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
  metrics["tool_calls"] = len((trace or {}).get("tool_calls", []))
  return metrics


def atif_from_trace(
  trace: dict[str, Any] | None, *, session_id: str | None, model: str, instruction: str,
) -> dict[str, Any]:
  """Map the portable Nonoka trace to Harbor's JSON ATIF interchange shape."""
  trace = trace or {}
  tool_calls = list(trace.get("tool_calls", []))
  calls_by_id = {str(call.get("id")): call for call in tool_calls if call.get("id") is not None}
  steps: list[dict[str, Any]] = [{
    "step_id": 1, "timestamp": trace.get("started_at"), "source": "user", "message": instruction,
  }]
  for index, turn in enumerate(trace.get("turns", []), start=2):
    response = turn.get("response", {}) if isinstance(turn, dict) else {}
    turn_calls = _calls_for_turn(response, calls_by_id)
    calls = [
      {
        "tool_call_id": str(call.get("id", "unknown")),
        "function_name": str(call.get("name", "unknown")),
        "arguments": call.get("arguments", {}),
      }
      for call in turn_calls
    ]
    observations = [
      {
        "source_call_id": str(call.get("id", "unknown")),
        "content": call.get("result") if "result" in call else call.get("error", {}),
      }
      for call in turn_calls
    ]
    usage = response.get("usage", {}) if isinstance(response, dict) else {}
    step: dict[str, Any] = {
      "step_id": index,
      "timestamp": turn.get("responded_at") or turn.get("requested_at"),
      "source": "agent",
      "model_name": model,
      "message": response.get("content", "") if isinstance(response, dict) else "",
      "metrics": {
        "prompt_tokens": int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or usage.get("output_tokens") or 0),
      },
    }
    if calls:
      step["tool_calls"] = calls
      step["observation"] = {"results": observations}
    steps.append(step)
  metrics = trace_metrics(trace)
  return {
    "schema_version": "ATIF-v1.7",
    "session_id": session_id,
    "agent": {"name": "nonoka", "model_name": model},
    "steps": steps,
    "final_metrics": {
      "total_prompt_tokens": metrics["input_tokens"],
      "total_completion_tokens": metrics["output_tokens"],
      "total_steps": len(steps),
    },
    "extra": {
      "nonoka_trace_schema": trace.get("schema_version"),
      "extensions": trace.get("extensions", []),
      "termination": trace.get("termination", {}),
    },
  }


def _calls_for_turn(response: Any, calls_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
  """Resolve trace calls from the IDs emitted in one model response.

  Nonoka records tool executions separately from model turns, so associating
  all calls with the first ATIF step loses the actual trajectory. The response
  is the authoritative parent of a tool call and stays stable across retries.
  """
  if not isinstance(response, dict):
    return []
  raw_calls = response.get("tool_calls") or []
  if not isinstance(raw_calls, list):
    return []
  selected: list[dict[str, Any]] = []
  for raw_call in raw_calls:
    if not isinstance(raw_call, dict):
      continue
    call_id = raw_call.get("id") or raw_call.get("tool_call_id")
    if call_id is not None and str(call_id) in calls_by_id:
      selected.append(calls_by_id[str(call_id)])
  return selected


def _terminal_result(value: Any) -> str:
  """Normalize Harbor environment command output without coupling to one backend."""
  if isinstance(value, str):
    return value
  if isinstance(value, dict):
    return json.dumps(value, ensure_ascii=False, default=str)
  payload = {
    key: getattr(value, key)
    for key in ("stdout", "stderr", "returncode", "return_code", "exit_code")
    if getattr(value, key, None) is not None
  }
  return json.dumps(payload or {"result": str(value)}, ensure_ascii=False, default=str)


def _terminal_observation(value: Any, max_chars: int = _DEFAULT_TERMINAL_OUTPUT_CHARS) -> ToolResponse:
  """Return a bounded, inspectable terminal observation to the model."""
  rendered = _terminal_result(value)
  try:
    payload = json.loads(rendered)
  except json.JSONDecodeError:
    payload = {"stdout": rendered}
  if not isinstance(payload, dict):
    payload = {"result": payload}

  omitted_chars = 0
  digests: dict[str, str] = {}
  for key in ("stdout", "stderr", "result"):
    text = payload.get(key)
    if not isinstance(text, str) or len(text) <= max_chars:
      continue
    omitted_chars += len(text) - max_chars
    digests[key] = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
    head = max_chars * 2 // 3
    payload[key] = text[:head] + "\n...[terminal output truncated]...\n" + text[-(max_chars - head):]

  if not omitted_chars:
    return ToolResponse(data=payload)
  return ToolResponse(
    data=payload,
    suggested_next_step="Terminal output was truncated; refine the next command with a path, grep pattern, head, or tail.",
    metadata={"truncated": True, "omitted_chars": omitted_chars, "sha256": digests},
  )


class NonokaHarborAgent(BaseAgent):
  """Current Harbor external agent for ``terminal-bench@2.0``."""

  def __init__(
    self,
    model_name: str | None = None,
    max_turns: int = 24,
    command_timeout_seconds: float = 180.0,
    max_terminal_output_chars: int = _DEFAULT_TERMINAL_OUTPUT_CHARS,
    requires_workspace_mutation: bool = False,
    max_exploration_turns: int = 3,
    temperature: float = 0.0,
    **kwargs: Any,
  ) -> None:
    if _HARBOR_IMPORT_ERROR is not None:
      raise RuntimeError("Nonoka Harbor adapter requires the optional `harbor` package.") from _HARBOR_IMPORT_ERROR
    super().__init__(**kwargs)
    self._model_name = model_name or str(kwargs.get("model") or "")
    self._max_turns = max_turns
    self._command_timeout_seconds = command_timeout_seconds
    self._max_terminal_output_chars = max(1_000, max_terminal_output_chars)
    self._requires_workspace_mutation = requires_workspace_mutation
    self._max_exploration_turns = max(1, max_exploration_turns)
    self._temperature = temperature

  @staticmethod
  def name() -> str:
    return "nonoka"

  def version(self) -> str | None:
    return "nonoka"

  async def setup(self, environment: Any) -> None:
    """No in-container installation: this adapter executes from the host."""
    _ = environment

  async def run(self, instruction: str, environment: Any, context: Any) -> None:
    @tool(execution=ToolExecution(stateful_action=True, mutates_workspace=True))
    async def execute_terminal(command: str, timeout_seconds: float | None = None) -> ToolResponse:
      """Execute a shell command in the Harbor task environment."""
      timeout = self._command_timeout_seconds if timeout_seconds is None else timeout_seconds
      if timeout <= 0 or timeout > self._command_timeout_seconds:
        raise ValueError(f"timeout_seconds must be between 0 and {self._command_timeout_seconds}")
      try:
        value = await environment.exec(command=command, timeout_sec=timeout)
      except TypeError:
        value = await environment.exec(command=command)
      return _terminal_observation(value, self._max_terminal_output_chars)

    extensions = (
      [WorkspaceProgressExtension(max_exploration_turns=self._max_exploration_turns)]
      if self._requires_workspace_mutation else []
    )

    agent = Agent(
      model=self._model_name,
      tools=[execute_terminal],
      system_prompt=_TERMINAL_SYSTEM_PROMPT,
      max_turns=self._max_turns,
      max_concurrency=1,
      temperature=self._temperature,
      default_timeout=self._command_timeout_seconds,
      extensions=extensions,
      metadata={"benchmark": "terminal-bench@2.0", "host": "harbor"},
    )
    result = await Runner(checkpoint="disabled", memory=None).run_react(agent, instruction, deps=None)
    metrics = trace_metrics(result.trace)
    for name, value in (
      ("n_input_tokens", metrics["input_tokens"]),
      ("n_output_tokens", metrics["output_tokens"]),
    ):
      try:
        setattr(context, name, value)
      except (AttributeError, TypeError, ValueError):
        pass
    self._write_atif(context, result.trace, getattr(result.session, "session_id", None), instruction)

  def _write_atif(self, context: Any, trace: dict[str, Any] | None, session_id: str | None, instruction: str) -> None:
    logs_dir = Path(getattr(self, "logs_dir", getattr(context, "logs_dir", Path.cwd())))
    logs_dir.mkdir(parents=True, exist_ok=True)
    path = logs_dir / "trajectory.json"
    path.write_text(
      json.dumps(atif_from_trace(trace, session_id=session_id, model=self._model_name, instruction=instruction), indent=2, default=str),
      encoding="utf-8",
    )
    try:
      setattr(context, "trajectory_path", str(path))
    except (AttributeError, TypeError, ValueError):
      pass
