from __future__ import annotations

"""
Shared execution utilities — ref resolution helpers used by all paradigms.

Scheduling *paradigms* (ReActAgent, PlanExecutor, ReflectiveAgent) live in
``nonoka.core.paradigm``.  This module only holds the low-level ``Ref``
resolution machinery that both ReAct and DAG execution need.
"""

import re
from typing import Any

from nonoka.core.plan import Ref


# --------------------------------------------------------------------------- #
# Ref resolution helpers
# --------------------------------------------------------------------------- #

def _resolve_path(data: Any, path: str) -> Any:
  """Resolve a dot-separated path (optionally with array indices) from *data*."""
  if not path:
    return data
  current = data
  for part in path.split("."):
    if current is None:
      return None

    # Handle bracket indexing: users[0], items[1][2]
    m = re.match(r"^([^\[]+)((?:\[\d+\])+)$", part)
    if m:
      key = m.group(1)
      indices = [int(x) for x in re.findall(r"\[(\d+)\]", m.group(2))]
      current = current[key] if isinstance(current, dict) else getattr(current, key, None)
      for idx in indices:
        if current is None:
          return None
        current = current[idx]
      continue

    if isinstance(current, dict):
      try:
        current = current[part]
      except KeyError:
        return None
    elif isinstance(current, list):
      try:
        idx = int(part)
        current = current[idx]
      except (ValueError, IndexError):
        return None
    else:
      current = getattr(current, part, None)
  return current


def _resolve_refs(data: Any, completed_steps: dict[str, Any]) -> Any:
  """Replace ``Ref`` markers with actual values from *completed_steps*.

  Recursively walks dicts and lists so refs nested at any depth are
  resolved (e.g. ``{"data": {"sum": ref("calc", "result")}}``).

  When a ref points to a step result that was normalised by
  ``unwrap_tool_response``, the wrapper dict
  ``{"result": X, "has_more": ...}`` is automatically unwrapped so the
  caller receives the raw *X* (only when *path* is empty – explicit paths
  such as ``ref("calc", "result")`` skip unwrapping).
  """
  if isinstance(data, Ref):
    source = completed_steps.get(data.step_id)
    if source is None:
      raise ValueError(
        f"Step '{data.step_id}' not found in completed_steps "
        f"(needed by ref)"
      )
    # source may be a StepResult (has .data) or raw dict
    source_data = source.data if hasattr(source, "data") else source
    resolved = _resolve_path(source_data, data.path)

    # If resolution failed and the step data is a normalised tool-response
    # wrapper, try resolving the path inside the wrapped original result.
    # This lets ``ref("step", "users.0.name")`` work when the tool returned
    # a dict like ``{"users": [...]}`` that was wrapped as
    # ``{"result": {"users": [...]}, "has_more": False}``.
    if resolved is None and isinstance(source_data, dict) and "result" in source_data and "has_more" in source_data:
      resolved = _resolve_path(source_data.get("result"), data.path)

    # Auto-unwrap normalised tool-response wrapper when no explicit path
    # was given.  This ensures that ``ref("calc")`` yields the original
    # tool return value (e.g. 56.0) rather than {"result": 56.0, ...}.
    if data.path == "" and isinstance(resolved, dict) and "result" in resolved and "has_more" in resolved:
      return resolved["result"]
    return resolved

  if isinstance(data, dict):
    return {k: _resolve_refs(v, completed_steps) for k, v in data.items()}

  if isinstance(data, list):
    return [_resolve_refs(item, completed_steps) for item in data]

  return data
