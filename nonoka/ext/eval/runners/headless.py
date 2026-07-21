"""Headless runner that executes a Nonoka Agent in a temporary workspace."""

from __future__ import annotations

import asyncio
import re
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from nonoka import Agent, Runner
from nonoka.ext.eval.checkers import CodeChecker, ToolUseChecker
from nonoka.ext.eval.models import EvalResult, EvalSample
from nonoka.ext.eval.runners.hooks import UsageHooks
from nonoka.ext.eval.tools import EvalDeps, get_eval_tools


class HeadlessEvalRunner:
  """Run samples using the normal ReAct loop and deterministic verifiers."""

  def __init__(
    self,
    model: str,
    max_turns: int = 8,
    timeout_seconds: float = 90.0,
    temperature: float | None = 0.0,
    runner_factory: Callable[[UsageHooks], Runner] | None = None,
  ) -> None:
    self.model = model
    self.max_turns = max_turns
    self.timeout_seconds = timeout_seconds
    self.temperature = temperature
    self._runner_factory = runner_factory

  async def evaluate(self, sample: EvalSample, *, baseline: bool = False) -> EvalResult:
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="nonoka-eval-") as temp_dir:
      root = Path(temp_dir)
      self._seed_workspace(root, sample)
      deps = EvalDeps(root)
      hooks = UsageHooks()
      agent = Agent(
        model=self.model,
        tools=[] if baseline else get_eval_tools(),
        system_prompt=self._system_prompt(sample, baseline),
        max_turns=1 if baseline else self.max_turns,
        max_steps=40,
        temperature=self.temperature,
      )
      runner = self._runner_factory(hooks) if self._runner_factory else Runner(
        checkpoint="memory", memory="in_memory", hooks=hooks,
      )
      try:
        result = await asyncio.wait_for(
          runner.run_react(agent, sample.prompt, deps=deps), timeout=self.timeout_seconds,
        )
      except TimeoutError:
        return EvalResult(
          sample_id=sample.id, success=False, runner_type="direct" if baseline else "agent",
          error="evaluation timed out", tool_trace=list(deps.tool_trace),
          metrics=self._metrics(None, hooks, time.monotonic() - start),
        )
      except Exception as exc:
        return EvalResult(
          sample_id=sample.id, success=False, runner_type="direct" if baseline else "agent",
          error=f"{type(exc).__name__}: {exc}", tool_trace=list(deps.tool_trace),
          metrics=self._metrics(None, hooks, time.monotonic() - start),
        )

      output = str(result.data or "")
      if sample.kind == "code" and not (root / "solution.py").exists():
        (root / "solution.py").write_text(_extract_code(output), encoding="utf-8")
      if sample.kind == "code":
        candidate_code = (root / "solution.py").read_text(encoding="utf-8")
        if sample.metadata.get("skip_local_verifier"):
          success, message = result.success, "candidate generated for an official external verifier"
        else:
          success, message = CodeChecker().check(sample, root)
      else:
        candidate_code = None
        success, message = ToolUseChecker().check(sample, root, deps)
      metrics = self._metrics(result.session, hooks, time.monotonic() - start)
      return EvalResult(
        sample_id=sample.id, success=success, runner_type="direct" if baseline else "agent",
        output=output, candidate_code=candidate_code, verifier_message=message,
        tool_trace=list(deps.tool_trace), metrics=metrics,
      )

  async def evaluate_many(self, samples: list[EvalSample], *, baseline: bool = False) -> list[EvalResult]:
    return [await self.evaluate(sample, baseline=baseline) for sample in samples]

  @staticmethod
  def _seed_workspace(root: Path, sample: EvalSample) -> None:
    for relative, content in sample.metadata.get("files", {}).items():
      path = root / relative
      path.parent.mkdir(parents=True, exist_ok=True)
      path.write_text(content, encoding="utf-8")

  @staticmethod
  def _system_prompt(sample: EvalSample, baseline: bool) -> str:
    if baseline:
      return (
        "Return only a complete Python implementation in a fenced code block. "
        "You have one response and no tools."
      )
    if sample.kind == "code":
      return (
        "Solve the programming task. Use write_file to create solution.py containing only "
        "the implementation, then briefly report completion."
      )
    return (
      "Complete the task using the provided workspace tools. Inspect files before editing, "
      "use execute_python to verify your result, and only then report completion."
    )

  @staticmethod
  def _metrics(session, hooks: UsageHooks, elapsed: float):
    metrics = hooks.metrics_for(session) if session is not None else __import__("nonoka.ext.eval.models", fromlist=["Metrics"]).Metrics()
    metrics.turns = int(getattr(session, "turn_count", 0)) if session is not None else 0
    metrics.wall_time_seconds = elapsed
    return metrics


def _extract_code(output: str) -> str:
  match = re.search(r"```(?:python)?\s*\n(.*?)```", output, flags=re.DOTALL | re.IGNORECASE)
  return match.group(1).strip() + "\n" if match else output.strip() + "\n"
