"""Tests for ToolResponse contract and normalization."""

import pytest
from nonoka.core.tool_response import (
  ToolResponse,
  make_tool_response,
  is_tool_response,
  unwrap_tool_response,
)


# --------------------------------------------------------------------------- #
# ToolResponse construction and serialization
# --------------------------------------------------------------------------- #

def test_tool_response_to_dict_includes_all_fields():
  """ToolResponse.to_dict should include all metadata fields."""
  tr = ToolResponse(
    data={"results": ["a", "b"]},
    has_more=True,
    next_cursor="abc123",
    prev_cursor="xyz789",
    total_count=100,
    suggested_next_step="Fetch next page",
    metadata={"source": "web"},
  )
  d = tr.to_dict()
  assert d["result"] == {"results": ["a", "b"]}
  assert d["has_more"] is True
  assert d["next_cursor"] == "abc123"
  assert d["prev_cursor"] == "xyz789"
  assert d["total_count"] == 100
  assert d["suggested_next_step"] == "Fetch next page"
  assert d["metadata"] == {"source": "web"}


def test_tool_response_to_dict_always_includes_has_more():
  """has_more should always be present so the LLM can distinguish false from unset."""
  tr = ToolResponse(data="simple result")
  d = tr.to_dict()
  assert d["result"] == "simple result"
  assert "has_more" in d
  assert d["has_more"] is False
  assert "next_cursor" not in d


def test_make_tool_response_convenience():
  """make_tool_response should build a ToolResponse correctly."""
  tr = make_tool_response(
    data=[1, 2, 3],
    has_more=True,
    next_cursor="page2",
    total_count=50,
    suggested_next_step="Continue",
    source="db",
  )
  assert tr.data == [1, 2, 3]
  assert tr.has_more is True
  assert tr.next_cursor == "page2"
  assert tr.total_count == 50
  assert tr.suggested_next_step == "Continue"
  assert tr.metadata == {"source": "db"}


# --------------------------------------------------------------------------- #
# unwrap_tool_response — the critical normalisation logic
# --------------------------------------------------------------------------- #

def test_unwrap_tool_response_expands_toolresponse():
  """A ToolResponse should be expanded into a dict with metadata."""
  tr = ToolResponse(data={"city": "Beijing"}, has_more=False)
  result = unwrap_tool_response(tr)
  assert isinstance(result, dict)
  assert result["result"] == {"city": "Beijing"}
  assert result["has_more"] is False


def test_unwrap_tool_response_wraps_plain_string():
  """A plain string should be wrapped in the standard shape."""
  result = unwrap_tool_response("hello")
  assert isinstance(result, dict)
  assert result["result"] == "hello"
  assert result["has_more"] is False


def test_unwrap_tool_response_wraps_plain_dict():
  """A plain dict should be wrapped in the standard shape."""
  result = unwrap_tool_response({"temperature": 25})
  assert isinstance(result, dict)
  assert result["result"] == {"temperature": 25}
  assert result["has_more"] is False


def test_unwrap_tool_response_wraps_list():
  """A plain list should be wrapped in the standard shape."""
  result = unwrap_tool_response([1, 2, 3])
  assert result["result"] == [1, 2, 3]
  assert result["has_more"] is False


def test_unwrap_tool_response_wraps_none():
  """None should be wrapped in the standard shape."""
  result = unwrap_tool_response(None)
  assert result["result"] is None
  assert result["has_more"] is False


def test_is_tool_response_true_for_toolresponse():
  assert is_tool_response(ToolResponse(data="x")) is True


def test_is_tool_response_false_for_plain_values():
  assert is_tool_response("hello") is False
  assert is_tool_response({"a": 1}) is False
  assert is_tool_response(42) is False


# --------------------------------------------------------------------------- #
# Partial ToolResponse shape (Bug fix: P3.1)
# --------------------------------------------------------------------------- #

def test_unwrap_tool_response_partial_shape_no_double_nesting():
  """A dict containing ``result`` but missing ``has_more`` should be
  treated as a partial ToolResponse shape and returned with ``has_more``
  filled in, rather than wrapped again (which would create
  ``{"result": {"result": 126}, "has_more": False}``)."""
  result = unwrap_tool_response({"result": 126})
  assert result == {"result": 126, "has_more": False}


def test_unwrap_tool_response_partial_shape_preserves_extra_fields():
  """A partial-shape dict should preserve any extra metadata fields."""
  result = unwrap_tool_response({"result": {"items": [1, 2]}, "total_count": 5})
  assert result["result"] == {"items": [1, 2]}
  assert result["has_more"] is False
  assert result["total_count"] == 5


def test_unwrap_tool_response_full_shape_unchanged():
  """A dict already containing both ``result`` and ``has_more`` should
  be returned exactly as-is."""
  original = {"result": {"items": [1, 2]}, "has_more": True, "next_cursor": "abc"}
  result = unwrap_tool_response(original)
  assert result is original
