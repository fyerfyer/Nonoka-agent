"""Typed, serialisable evaluation records."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class Metrics(BaseModel):
  turns: int = 0
  llm_calls: int = 0
  tool_calls: int = 0
  input_tokens: int = 0
  output_tokens: int = 0
  estimated_cost_usd: float | None = None
  wall_time_seconds: float = 0.0

  @property
  def total_tokens(self) -> int:
    return self.input_tokens + self.output_tokens


class EvalSample(BaseModel):
  id: str
  dataset: str
  prompt: str
  kind: str = "code"
  metadata: dict[str, Any] = Field(default_factory=dict)


class EvalResult(BaseModel):
  sample_id: str
  success: bool
  runner_type: str = "agent"
  strategy: str = "tool_assisted"
  output: str = ""
  candidate_code: str | None = None
  error: str | None = None
  verifier_message: str = ""
  tool_trace: list[str] = Field(default_factory=list)
  trace: dict[str, Any] | None = None
  metrics: Metrics = Field(default_factory=Metrics)


class EvalRun(BaseModel):
  run_id: str = Field(default_factory=lambda: uuid4().hex[:12])
  timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
  dataset: str
  model: str
  runner_type: str = "paired-headless"
  source: str = "builtin"
  samples: list[EvalResult] = Field(default_factory=list)
  baseline_samples: list[EvalResult] = Field(default_factory=list)
  metadata: dict[str, Any] = Field(default_factory=dict)

  def summary(self) -> dict[str, Any]:
    total = len(self.samples)
    passed = sum(sample.success for sample in self.samples)
    metrics = [sample.metrics for sample in self.samples]
    baseline_passed = sum(sample.success for sample in self.baseline_samples)
    return {
      "run_id": self.run_id,
      "dataset": self.dataset,
      "model": self.model,
      "runner_type": self.runner_type,
      "source": self.source,
      "total": total,
      "passed": passed,
      "failed": total - passed,
      "pass_at_1": passed / total if total else 0.0,
      "avg_turns": sum(m.turns for m in metrics) / total if total else 0.0,
      "avg_llm_calls": sum(m.llm_calls for m in metrics) / total if total else 0.0,
      "avg_tool_calls": sum(m.tool_calls for m in metrics) / total if total else 0.0,
      "total_tokens": sum(m.total_tokens for m in metrics),
      "total_estimated_cost_usd": _sum_cost(m.estimated_cost_usd for m in metrics),
      "direct_passed": baseline_passed if self.baseline_samples else None,
      "direct_pass_at_1": baseline_passed / total if self.baseline_samples and total else None,
      "agent_lift": (passed - baseline_passed) / total if self.baseline_samples and total else None,
    }

  def write(self, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(self.model_dump_json(indent=2), encoding="utf-8")


def _sum_cost(costs: Any) -> float | None:
  values = list(costs)
  if any(value is None for value in values):
    return None
  return float(sum(values))


class Leaderboard(BaseModel):
  entries: list[dict[str, Any]] = Field(default_factory=list)

  @classmethod
  def load(cls, path: Path) -> "Leaderboard":
    if not path.exists():
      return cls()
    return cls.model_validate_json(path.read_text(encoding="utf-8"))

  def append(self, run: EvalRun) -> None:
    summary = run.summary()
    summary["timestamp"] = run.timestamp.isoformat()
    self.entries.append(summary)

  def save(self, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
