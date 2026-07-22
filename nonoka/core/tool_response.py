"""
Standardized tool response contract for Nonoka agents.

Tools can return plain values (backward-compatible), or use ``ToolResponse``
to communicate richer metadata to the Agent loop:

* ``has_more`` — tells the Agent whether additional data is available.
* ``next_cursor`` / ``prev_cursor`` — pagination cursors.
* ``total_count`` — total result count (useful for search/list tools).
* ``suggested_next_step`` — a hint from the tool about what to do next.

Usage::

    from nonoka.core.tool_response import ToolResponse, make_tool_response

    @tool
    async def search_web(ctx: RunContext, query: str, cursor: str | None = None) -> ToolResponse:
        results, next_cursor = await _do_search(query, cursor)
        return ToolResponse(
            data={"results": results, "query": query},
            has_more=next_cursor is not None,
            next_cursor=next_cursor,
            suggested_next_step="Summarise the findings and stop searching."
            if len(results) >= 5 else "Refine query and search again.",
        )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResponse:
  """Rich tool response with metadata for Agent loop decision-making.

  Fields:
    data: The actual payload returned by the tool (any JSON-serializable value).
    has_more: Whether there is additional data available (e.g. paginated search).
    next_cursor: Opaque cursor string for fetching the next page.
    prev_cursor: Opaque cursor string for fetching the previous page.
    total_count: Total number of items available across all pages.
    suggested_next_step: Human-readable hint from the tool to guide the LLM.
    metadata: Arbitrary extra metadata for debugging or observability.
  """

  data: Any
  has_more: bool = False
  next_cursor: str | None = None
  prev_cursor: str | None = None
  total_count: int | None = None
  suggested_next_step: str | None = None
  progress: bool | None = None
  metadata: dict[str, Any] = field(default_factory=dict)

  def to_dict(self) -> dict[str, Any]:
    """Serialize to a plain dict for LLM consumption."""
    result: dict[str, Any] = {
      "result": self.data,
      "has_more": self.has_more,
    }
    if self.next_cursor is not None:
      result["next_cursor"] = self.next_cursor
    if self.prev_cursor is not None:
      result["prev_cursor"] = self.prev_cursor
    if self.total_count is not None:
      result["total_count"] = self.total_count
    if self.suggested_next_step is not None:
      result["suggested_next_step"] = self.suggested_next_step
    if self.progress is not None:
      result["progress"] = self.progress
    if self.metadata:
      result["metadata"] = self.metadata
    return result


# --------------------------------------------------------------------------- #
# Convenience helpers
# --------------------------------------------------------------------------- #

def make_tool_response(
  data: Any,
  *,
  has_more: bool = False,
  next_cursor: str | None = None,
  total_count: int | None = None,
  suggested_next_step: str | None = None,
  progress: bool | None = None,
  **metadata: Any,
) -> ToolResponse:
  """Create a ``ToolResponse`` with common fields."""
  return ToolResponse(
    data=data,
    has_more=has_more,
    next_cursor=next_cursor,
    total_count=total_count,
    suggested_next_step=suggested_next_step,
    progress=progress,
    metadata=metadata,
  )


def is_tool_response(value: Any) -> bool:
  """Return True if *value* is a ``ToolResponse`` instance."""
  return isinstance(value, ToolResponse)


def unwrap_tool_response(value: Any) -> dict[str, Any]:
  """Normalise any tool return value into the standard response shape.

  * ``ToolResponse`` → expanded dict with metadata.
  * Dict already containing ``result`` + ``has_more`` → returned as-is.
  * Dict containing ``result`` but missing ``has_more`` → ``has_more`` is
    filled in as ``False`` and the dict is returned without double-nesting.
  * Plain value → ``{"result": value, "has_more": false}``.

  This ensures the LLM always sees a consistent format regardless of
  whether the tool author used ``ToolResponse``, returned a raw dict
  in the standard shape, or returned a plain value.
  """
  if isinstance(value, ToolResponse):
    return value.to_dict()
  if isinstance(value, dict) and "result" in value and "has_more" in value:
    return value
  # If a dict already contains ``result`` but is missing ``has_more``,
  # treat it as a partial ToolResponse shape and fill in the missing field
  # rather than wrapping it again (which would create double nesting).
  if isinstance(value, dict) and "result" in value:
    return {**value, "has_more": value.get("has_more", False)}
  return {"result": value, "has_more": False}
