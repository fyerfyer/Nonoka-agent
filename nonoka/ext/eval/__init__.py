"""Reproducible evaluation utilities for Nonoka agents.

The public command is ``python -m nonoka.ext.eval``.  The framework owns
dataset semantics, verifiers, and result artifacts; frontends only delegate
arguments to this module.
"""

from typing import TYPE_CHECKING

from nonoka.ext.eval.models import EvalResult, EvalRun, EvalSample, Leaderboard, Metrics

if TYPE_CHECKING:
  from nonoka.ext.eval.runners.headless import HeadlessEvalRunner

__all__ = ["EvalResult", "EvalRun", "EvalSample", "HeadlessEvalRunner", "Leaderboard", "Metrics"]


def __getattr__(name: str):
  """Avoid booting the LLM stack for metadata-only eval commands."""
  if name == "HeadlessEvalRunner":
    from nonoka.ext.eval.runners.headless import HeadlessEvalRunner
    return HeadlessEvalRunner
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
