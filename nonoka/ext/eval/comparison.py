"""Repeated, paired evaluation experiments for explicit coding strategies."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from nonoka.ext.eval.models import EvalRun, EvalSample, StrategyComparison
from nonoka.ext.eval.runners.headless import HeadlessEvalRunner


async def compare_strategies(
  samples: Iterable[EvalSample],
  *,
  model: str,
  strategies: tuple[str, ...] = ("direct", "tool_assisted", "verified_repair"),
  trials: int = 3,
  max_turns: int = 8,
  timeout_seconds: float = 90.0,
  temperature: float | None = 0.0,
  runner_factory: Any | None = None,
) -> StrategyComparison:
  """Run every strategy against the identical ordered sample IDs each trial."""
  frozen_samples = list(samples)
  if trials < 1:
    raise ValueError("trials must be at least 1")
  valid = {"direct", "tool_assisted", "verified_repair"}
  unknown = set(strategies) - valid
  if unknown:
    raise ValueError(f"unknown strategies: {', '.join(sorted(unknown))}")
  comparison = StrategyComparison(
    dataset=frozen_samples[0].dataset if frozen_samples else "unknown",
    model=model, sample_ids=[sample.id for sample in frozen_samples], trials=trials,
    metadata={"temperature": temperature, "max_turns": max_turns, "timeout_seconds": timeout_seconds},
  )
  for trial in range(1, trials + 1):
    for strategy in strategies:
      runner = HeadlessEvalRunner(
        model, max_turns=max_turns, timeout_seconds=timeout_seconds,
        temperature=temperature, strategy=strategy, runner_factory=runner_factory,
      )
      results = await runner.evaluate_many(frozen_samples)
      comparison.runs.append(EvalRun(
        dataset=comparison.dataset, model=model, samples=results,
        source="strategy-comparison", metadata={"strategy": strategy, "trial": trial},
      ))
  return comparison
