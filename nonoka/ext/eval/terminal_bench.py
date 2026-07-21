"""Official Terminal-Bench custom-agent adapter.

This module deliberately lives in :mod:`nonoka.ext.eval`: Terminal-Bench owns
the Docker lifecycle, task images, and verifier. Import it only from an
evaluation environment that has the optional ``terminal-bench`` package.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from nonoka import Agent, Runner
from nonoka.core.tool import tool

try:  # Terminal-Bench is intentionally not a core dependency.
  from terminal_bench.agents.base_agent import AgentResult, BaseAgent
  from terminal_bench.agents.failure_mode import FailureMode
  from terminal_bench.terminal.tmux_session import TmuxSession
except ImportError as exc:  # pragma: no cover - exercised in the harness environment
  _TERMINAL_BENCH_IMPORT_ERROR: ImportError | None = exc
  AgentResult = Any  # type: ignore[misc,assignment]
  BaseAgent = object  # type: ignore[misc,assignment]
  FailureMode = Any  # type: ignore[misc,assignment]
  TmuxSession = Any  # type: ignore[misc,assignment]
else:
  _TERMINAL_BENCH_IMPORT_ERROR = None


def prepare_terminal_session(session: Any, timeout_seconds: float) -> None:
  """Make a verifier-owned terminal safe for one-command tool execution.

  Terminal-Bench gives the agent an interactive tmux shell.  Commands such as
  ``git log`` and ``less`` otherwise enter a pager and never reach the
  completion marker appended by ``TmuxSession``.  These settings only affect
  the task shell; they do not alter the host process or the task files.
  """
  session.send_keys(
    keys=[
      "export PAGER=cat GIT_PAGER=cat GIT_TERMINAL_PROMPT=0 LESS=-FRSX",
      "Enter",
    ],
    block=True,
    max_timeout_sec=timeout_seconds,
  )


class NonokaTerminalBenchAgent(BaseAgent):
  """Run one Nonoka ReAct session against an official task terminal."""

  def __init__(
    self,
    model_name: str,
    max_turns: int = 24,
    command_timeout_seconds: float = 180.0,
    temperature: float = 0.0,
    **kwargs: Any,
  ) -> None:
    if _TERMINAL_BENCH_IMPORT_ERROR is not None:
      raise RuntimeError(
        "Nonoka Terminal-Bench adapter requires `terminal-bench` in the "
        "same evaluation environment. Install it in a dedicated venv."
      ) from _TERMINAL_BENCH_IMPORT_ERROR
    super().__init__(**kwargs)
    self._model_name = model_name
    self._max_turns = max_turns
    self._command_timeout_seconds = command_timeout_seconds
    self._temperature = temperature

  @staticmethod
  def name() -> str:
    return "nonoka"

  def perform_task(
    self,
    instruction: str,
    session: TmuxSession,
    logging_dir: Path | None = None,
  ) -> AgentResult:
    """Let Nonoka solve one task through Terminal-Bench's tmux session."""
    if _TERMINAL_BENCH_IMPORT_ERROR is not None:
      raise RuntimeError("Terminal-Bench is not installed in this environment.")

    prepare_terminal_session(session, self._command_timeout_seconds)

    @tool
    async def read_terminal() -> str:
      """Read the complete current terminal buffer before deciding what to do."""
      return session.capture_pane(capture_entire=True)

    @tool
    async def execute_terminal(command: str, timeout_seconds: float | None = None) -> str:
      """Run one shell command in the task terminal and return new output."""
      if "\x00" in command or "\n" in command or "\r" in command:
        raise ValueError("execute_terminal accepts exactly one shell command")
      timeout = self._command_timeout_seconds if timeout_seconds is None else timeout_seconds
      if timeout <= 0 or timeout > self._command_timeout_seconds:
        raise ValueError(
          f"timeout_seconds must be between 0 and {self._command_timeout_seconds}"
        )
      session.send_keys(
        keys=[command, "Enter"], block=True, max_timeout_sec=timeout,
      )
      return session.get_incremental_output()

    agent = Agent(
      model=self._model_name,
      tools=[read_terminal, execute_terminal],
      system_prompt=(
        "You are solving a Terminal-Bench task in a Linux terminal. Inspect "
        "the terminal before acting. Work only through the provided tools, "
        "verify the requested result, and stop when the task is complete."
      ),
      max_turns=self._max_turns,
      max_concurrency=1,
      temperature=self._temperature,
      default_timeout=self._command_timeout_seconds,
      metadata={"benchmark": "terminal-bench"},
    )
    result = asyncio.run(
      Runner(checkpoint="disabled", memory=None).run_react(agent, instruction, deps=None)
    )
    if logging_dir is not None:
      logging_dir.mkdir(parents=True, exist_ok=True)
      (logging_dir / "nonoka-result.json").write_text(
        json.dumps({"success": result.success, "data": result.data, "error": result.error}, default=str),
        encoding="utf-8",
      )
    return AgentResult(
      failure_mode=FailureMode.NONE if result.success else FailureMode.UNKNOWN_AGENT_ERROR,
      timestamped_markers=[
        (session.get_asciinema_timestamp(), "nonoka: completed" if result.success else "nonoka: failed")
      ],
    )
