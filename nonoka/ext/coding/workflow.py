"""First-class, explicit coding workflow assembly."""

from __future__ import annotations

import inspect
import shlex
from dataclasses import dataclass
from typing import Any, Callable

from nonoka import Agent
from nonoka.core.types import RunResult

from .extensions import VerifierRepairExtension
from .strategy import CodeStrategy, CodeStrategyRouter
from .diagnostics import VerificationReport, VerifierDiagnostic, VerifierDiagnosticCode


class TerminalCommandEvaluator:
  """Run a caller-approved terminal verifier and normalize its result.

  ``execute`` is deliberately injected: the evaluator can target a local
  workspace, a remote terminal, or a Harbor environment without granting the
  coding extension any ambient shell access. The command is never inferred
  from task text.
  """

  def __init__(
    self,
    verify_command: tuple[str, ...],
    execute: Callable[[tuple[str, ...]], Any],
  ) -> None:
    if not verify_command:
      raise ValueError("verify_command must not be empty")
    self.verify_command = verify_command
    self._execute = execute

  async def evaluate(self, _result: RunResult) -> Any:
    try:
      outcome = self._execute(self.verify_command)
      if inspect.isawaitable(outcome):
        outcome = await outcome
    except TimeoutError:
      return VerificationReport(
        False, "verifier command timed out",
        VerifierDiagnostic(VerifierDiagnosticCode.TIMEOUT, "verifier command timed out"),
      )
    except Exception as exc:
      message = f"verifier command failed to execute: {type(exc).__name__}: {exc}"
      return VerificationReport(
        False, message, VerifierDiagnostic(VerifierDiagnosticCode.PROCESS_ERROR, message),
      )
    exit_code, output = _terminal_verifier_outcome(outcome)
    if exit_code == 0:
      return VerificationReport(True, "verifier command passed")
    command = shlex.join(self.verify_command)
    message = f"verifier command exited {exit_code}: {command}"
    if output:
      message += f"\n{output[-2000:]}"
    return VerificationReport(
      False, message, VerifierDiagnostic(VerifierDiagnosticCode.TEST_FAILURE, message),
    )


def _terminal_verifier_outcome(outcome: Any) -> tuple[int, str]:
  """Accept common process-result shapes without binding to a terminal SDK."""
  if isinstance(outcome, int):
    return outcome, ""
  if isinstance(outcome, tuple) and len(outcome) == 2:
    return int(outcome[0]), str(outcome[1] or "")
  if isinstance(outcome, dict):
    code = outcome.get("exit_code", outcome.get("returncode", 1))
    return int(code), "\n".join(str(outcome.get(key) or "") for key in ("stdout", "stderr"))
  code = getattr(outcome, "exit_code", getattr(outcome, "returncode", 1))
  return int(code), "\n".join(str(getattr(outcome, key, "") or "") for key in ("stdout", "stderr"))


@dataclass(frozen=True)
class CodingWorkflow:
  """Build a coding agent without manually wiring router and extensions.

  The caller supplies known task capabilities; the workflow never infers an
  expensive strategy from prompt text.  ``strategy=None`` applies the safe
  router policy: direct without a workspace, tool-assisted with a workspace,
  and verified repair only when both a workspace and evaluator exist.
  """

  requires_workspace: bool = False
  evaluator: Any | None = None
  strategy: CodeStrategy | None = None
  max_repairs: int = 2

  def resolve_strategy(self) -> CodeStrategy:
    if self.strategy is not None:
      if self.strategy is CodeStrategy.VERIFIED_REPAIR and self.evaluator is None:
        raise ValueError("verified_repair requires a deterministic evaluator")
      return self.strategy
    return CodeStrategyRouter().choose(
      deterministic_verifier=self.evaluator is not None,
      requires_workspace=self.requires_workspace,
    )

  def extensions(self) -> list[Any]:
    if self.resolve_strategy() is CodeStrategy.VERIFIED_REPAIR:
      return [VerifierRepairExtension(self.evaluator, max_repairs=max(0, self.max_repairs))]
    return []

  def build_agent(
    self,
    *,
    model: str,
    tools: list[Any] | None = None,
    system_prompt: str = "",
    **agent_options: Any,
  ) -> Agent:
    strategy = self.resolve_strategy()
    return Agent(
      model=model,
      tools=[] if strategy is CodeStrategy.DIRECT else list(tools or []),
      system_prompt=system_prompt,
      extensions=self.extensions(),
      **agent_options,
    )

  async def run(
    self,
    runner: Any,
    *,
    model: str,
    prompt: str,
    deps: Any,
    tools: list[Any] | None = None,
    system_prompt: str = "",
    **agent_options: Any,
  ) -> RunResult:
    agent = self.build_agent(
      model=model, tools=tools, system_prompt=system_prompt, **agent_options,
    )
    return await runner.run_react(agent, prompt, deps=deps)


@dataclass(frozen=True)
class TerminalCodingWorkflow(CodingWorkflow):
  """Coding workflow that only repairs when a task supplies a verifier command.

  Terminal-Bench hidden verifiers remain outside the agent environment.  A
  caller that wants bounded in-session repair must provide an evaluator built
  from an explicit, task-local ``verify_command``.
  """

  verify_command: tuple[str, ...] | None = None

  def resolve_strategy(self) -> CodeStrategy:
    strategy = super().resolve_strategy()
    if strategy is CodeStrategy.VERIFIED_REPAIR and not self.verify_command:
      raise ValueError("Terminal verified_repair requires an explicit verify_command")
    return strategy
