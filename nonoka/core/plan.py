from dataclasses import dataclass, field
from typing import FrozenSet

from nonoka.core.types import RetryPolicy


@dataclass(frozen=True)
class Step:
  """Immutable single execution step"""
  id: str
  tool: str                                 # Reference to the tool name
  args: dict = field(default_factory=dict)  # Parameters or references to other steps
  depends_on: FrozenSet[str] = frozenset()  # Step ID dependencies
  retry: RetryPolicy = field(default_factory=RetryPolicy)
  timeout: float | None = None              # None indicates fallback to Agent's default timeout


@dataclass(frozen=True)
class Plan:
  """Immutable execution plan (DAG)"""
  steps: tuple[Step, ...]
  objective: str = ""
  metadata: dict = field(default_factory=dict)

  def topological_groups(self) -> list[list[str]]:
    """
    Perform topological sorting, grouping step IDs by layers for subsequent parallel execution by DAGScheduler.
    """
    # TODO: Implement DAG topological sorting algorithm, you can write the logic later
    # This stage can be left as is, or NotImplementedError can be raised
    raise NotImplementedError("Topological sorting to be implemented")

  def get_step(self, step_id: str) -> Step | None:
    for step in self.steps:
      if step.id == step_id:
        return step
    return None