from __future__ import annotations

from collections.abc import Callable

from nonoka.ext.eval.models import EvalSample

DatasetLoader = Callable[[int | None], list[EvalSample]]


class DatasetLoaderError(RuntimeError):
  """Raised when a requested open dataset cannot be loaded."""
