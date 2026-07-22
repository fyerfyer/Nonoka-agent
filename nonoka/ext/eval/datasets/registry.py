from __future__ import annotations

from dataclasses import dataclass

from nonoka.ext.eval.datasets.base import DatasetLoader, DatasetLoaderError
from nonoka.ext.eval.datasets.builtins import load_humaneval, load_mbpp, load_tool_use
from nonoka.ext.eval.datasets.complex_mbpp import load_complex_mbpp_v1
from nonoka.ext.eval.models import EvalSample


@dataclass(frozen=True)
class DatasetDefinition:
  name: str
  description: str
  source: str
  loader: DatasetLoader


class DatasetRegistry:
  def __init__(self) -> None:
    self._datasets: dict[str, DatasetDefinition] = {}

  def register(self, definition: DatasetDefinition) -> None:
    self._datasets[definition.name] = definition

  def list(self) -> list[DatasetDefinition]:
    return list(self._datasets.values())

  def load(self, name: str, limit: int | None = None, offset: int = 0) -> list[EvalSample]:
    if offset < 0:
      raise DatasetLoaderError("offset must be zero or greater")
    try:
      samples = self._datasets[name].loader(None)
    except KeyError as exc:
      available = ", ".join(sorted(self._datasets))
      raise DatasetLoaderError(f"Unknown dataset '{name}'. Available: {available}") from exc
    end = None if limit is None else offset + limit
    return samples[offset:end]


_REGISTRY = DatasetRegistry()
_REGISTRY.register(DatasetDefinition(
  "humaneval", "OpenAI HumanEval functional code-generation tasks (official dataset loader).",
  "openai/openai_humaneval", load_humaneval,
))
_REGISTRY.register(DatasetDefinition(
  "mbpp", "Google MBPP sanitized code-generation tasks (official dataset loader).",
  "google-research-datasets/mbpp", load_mbpp,
))
_REGISTRY.register(DatasetDefinition(
  "mbpp-complex-v1", "Versioned 20-task complex slice of sanitized MBPP for paired strategy comparisons.",
  "google-research-datasets/mbpp (fixture mbpp-complex-v1)", load_complex_mbpp_v1,
))
_REGISTRY.register(DatasetDefinition(
  "tool_use", "Bundled deterministic filesystem/tool-use smoke and regression tasks (not a public benchmark).",
  "nonoka bundled v1", load_tool_use,
))


def get_registry() -> DatasetRegistry:
  return _REGISTRY
