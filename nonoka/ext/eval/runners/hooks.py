"""Hooks that turn framework events into benchmark metrics."""

from __future__ import annotations

from collections import defaultdict

from nonoka.core.hooks import Hooks
from nonoka.core.llm import LLMResponse
from nonoka.ext.eval.models import Metrics


class UsageHooks(Hooks):
  def __init__(self) -> None:
    super().__init__()
    self._metrics: dict[str, Metrics] = defaultdict(Metrics)

  async def on_llm_response(self, ctx, response: LLMResponse) -> None:
    metrics = self._metrics[ctx.session.session_id]
    metrics.llm_calls += 1
    usage = response.usage or {}
    metrics.input_tokens += int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    metrics.output_tokens += int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)

  async def on_tool_start(self, ctx, tool_name: str, arguments: dict) -> None:
    self._metrics[ctx.session.session_id].tool_calls += 1

  def metrics_for(self, session) -> Metrics:
    metrics = self._metrics.get(session.session_id, Metrics())
    return metrics.model_copy(deep=True)
