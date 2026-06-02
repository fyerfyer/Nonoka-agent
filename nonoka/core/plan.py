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
    groups: list[list[str]] = []
    in_degree = {step.id: 0 for step in self.steps}
    graph: dict[str, list[str]] = {step.id: [] for step in self.steps}
    
    for step in self.steps:
      for dep in step.depends_on:
        if dep in graph:
          graph[dep].append(step.id)
          in_degree[step.id] += 1

    queue = [node for node, degree in in_degree.items() if degree == 0]
    
    while queue:
      groups.append(queue)
      next_queue = []
      for node in queue:
        for neighbor in graph[node]:
          in_degree[neighbor] -= 1
          if in_degree[neighbor] == 0:
            next_queue.append(neighbor)
      queue = next_queue
        
    if sum(len(group) for group in groups) != len(self.steps):
      raise ValueError("Plan contains a cycle in dependencies")
        
    return groups

  def get_step(self, step_id: str) -> Step | None:
    for step in self.steps:
      if step.id == step_id:
        return step
    return None