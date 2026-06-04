from dataclasses import dataclass, field
from typing import FrozenSet, Any
from collections.abc import Callable

from nonoka.core.types import RetryPolicy


class Ref:
  """Reference marker used in PlanBuilder step arguments.

  At build time ``Ref`` is just a placeholder.  At execution time the
  DAGScheduler resolves it to the actual value from a previously completed
  step.

  Example::

      .step("analyze", analyze_deps, repo=ref("fetch", "result"))
  """

  def __init__(self, step_id: str, path: str = ""):
    self.step_id = step_id
    self.path = path

  def __repr__(self) -> str:  # pragma: no cover
    return f"ref({self.step_id!r}, {self.path!r})"

  def __eq__(self, other: object) -> bool:
    if not isinstance(other, Ref):
      return NotImplemented
    return self.step_id == other.step_id and self.path == other.path

  def __hash__(self) -> int:
    return hash((self.step_id, self.path))


def ref(step_id: str, path: str | None = None) -> Ref:
  """Create a reference to the output of a previous step.

  Supports two calling conventions:

  * **Explicit** — ``ref("fetch")`` or ``ref("fetch", "data.users[0]")``
  * **Shorthand** — ``ref("fetch.result")`` (first dot splits *step_id* from *path*)

  If *path* is omitted and *step_id* contains a dot, the first segment is
  treated as the step ID and the remainder as the path.  If *step_id*
  contains no dot and *path* is omitted, the step result is returned as-is
  (equivalent to ``session.completed_steps[step_id].data``).

  Args:
    step_id: The ID of the step whose output should be referenced.
    path: Dot-separated path into the step result.  Default is empty,
      meaning the raw step data is returned.  Nested paths such as
      ``"users[0].id"`` are supported.
  """
  if path is not None:
    return Ref(step_id, path)

  if "." in step_id:
    parts = step_id.split(".", 1)
    return Ref(parts[0], parts[1])

  return Ref(step_id, "")


@dataclass(frozen=True)
class Step:
  """Immutable single execution step"""
  id: str
  tool: str                                 # Reference to the tool name
  args: dict = field(default_factory=dict)  # Parameters or references to other steps
  depends_on: FrozenSet[str] = frozenset()  # Step ID dependencies
  retry: RetryPolicy = field(default_factory=RetryPolicy)
  timeout: float | None = None              # None indicates fallback to Agent's default timeout
  idempotent: bool = False                  # If True, step may be safely skipped on resume
  cache_key: str | None = None              # Custom key for external cache lookup
  force_rerun: bool = False                 # If True, always re-execute even if completed


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


class PlanBuilder:
  """Chainable API for constructing immutable ``Plan`` objects.

  Usage::

    plan = (
      PlanBuilder(objective="Deploy app")
      .step("build", build_image, tag="v1.0")
      .step("test", run_tests, image=ref("build", "result"))
      .step("deploy", deploy_app, image=ref("build", "result"), tests=ref("test", "result"))
      .build()
    )

  Dependencies are **auto-detected** from ``ref()`` markers in step arguments,
  so callers rarely need to pass ``depends_on`` explicitly.
  """

  def __init__(self, objective: str = ""):
    self.objective = objective
    self._steps: list[Step] = []
    self._step_ids: set[str] = set()

  def step(
    self,
    id: str,
    tool: str | Callable[..., Any],
    depends_on: set[str] | list[str] | str | None = None,
    retry: RetryPolicy | None = None,
    timeout: float | None = None,
    idempotent: bool = False,
    cache_key: str | None = None,
    force_rerun: bool = False,
    **args: Any,
  ) -> "PlanBuilder":
    """Add a step to the plan.

    Args:
      id: Unique step identifier.
      tool: Tool name (``str``) or a callable decorated with ``@tool`` (has
        a ``name`` attribute).
      depends_on: Explicit step ID dependencies.  Can be a single string,
        a list of strings, or a set of strings.  In addition, any ``Ref``
        value in ``**args`` is automatically detected as a dependency.
      retry: Per-step retry policy.  Overrides the Agent's default.
      timeout: Per-step timeout in seconds.  Overrides the Agent's default.
      idempotent: Whether the step can be safely skipped if already completed.
      cache_key: Optional custom key for external cache lookup.
      force_rerun: If True, always re-execute even if already completed.
      **args: Static arguments or ``ref()`` markers.  Any ``Ref`` value
        automatically adds the referenced step to ``depends_on``.
    """
    if id in self._step_ids:
      raise ValueError(f"Duplicate step id: {id}")

    # Resolve tool name — accept str, @tool-decorated callables, or Capability instances
    if hasattr(tool, "name"):
      tool_name = tool.name
    elif isinstance(tool, str):
      tool_name = tool
    else:
      raise TypeError(f"tool must be a string name or a @tool-decorated callable, got {type(tool)}")

    # Collect explicit dependencies
    deps: set[str] = set()
    if depends_on is not None:
      if isinstance(depends_on, str):
        deps.add(depends_on)
      else:
        deps.update(depends_on)

    # Auto-detect dependencies from Ref values (recursively in lists/dicts)
    def _collect_refs(obj: Any) -> None:
      if isinstance(obj, Ref):
        deps.add(obj.step_id)
      elif isinstance(obj, dict):
        for v in obj.values():
          _collect_refs(v)
      elif isinstance(obj, list):
        for item in obj:
          _collect_refs(item)

    for val in args.values():
      _collect_refs(val)

    step = Step(
      id=id,
      tool=tool_name,
      args=args,
      depends_on=frozenset(deps),
      retry=retry if retry else RetryPolicy(),
      timeout=timeout,
      idempotent=idempotent,
      cache_key=cache_key,
      force_rerun=force_rerun,
    )
    self._steps.append(step)
    self._step_ids.add(id)
    return self

  def build(self) -> Plan:
    """Build and return the immutable ``Plan``."""
    return Plan(steps=tuple(self._steps), objective=self.objective)