"""Tests for Ref resolution and serialization."""

import pytest
from pydantic import BaseModel

from nonoka.core.plan import Ref, ref
from nonoka.core.scheduler import _resolve_refs, _resolve_path


# --------------------------------------------------------------------------- #
# Ref construction
# --------------------------------------------------------------------------- #

def test_ref_explicit_args():
  """ref() with explicit step_id and path."""
  r = ref("fetch", "result")
  assert r.step_id == "fetch"
  assert r.path == "result"


def test_ref_shorthand():
  """ref() with dot shorthand."""
  r = ref("fetch.result")
  assert r.step_id == "fetch"
  assert r.path == "result"


def test_ref_no_path():
  """ref() with just step_id."""
  r = ref("fetch")
  assert r.step_id == "fetch"
  assert r.path == ""


def test_ref_is_pydantic_model():
  """Ref should be a Pydantic BaseModel for serialization."""
  r = ref("step1", "data")
  assert isinstance(r, BaseModel)
  # frozen = hashable
  d = {r: "value"}
  assert d[r] == "value"


# --------------------------------------------------------------------------- #
# _resolve_path
# --------------------------------------------------------------------------- #

def test_resolve_path_dict_key():
  """Resolve simple dict key path."""
  data = {"user": {"name": "Alice"}}
  assert _resolve_path(data, "user.name") == "Alice"


def test_resolve_path_list_index():
  """Resolve list index in path."""
  data = {"users": [{"name": "Alice"}, {"name": "Bob"}]}
  assert _resolve_path(data, "users.0.name") == "Alice"
  assert _resolve_path(data, "users.1.name") == "Bob"


def test_resolve_path_empty():
  """Empty path returns data as-is."""
  data = {"result": 42}
  assert _resolve_path(data, "") == data


def test_resolve_path_missing():
  """Missing path returns None."""
  data = {"user": {"name": "Alice"}}
  assert _resolve_path(data, "user.age") is None


# --------------------------------------------------------------------------- #
# _resolve_refs auto-unwrap
# --------------------------------------------------------------------------- #

class FakeStepResult:
  """Mock StepResult for testing."""
  def __init__(self, data):
    self.data = data


def test_resolve_refs_unwraps_normalised_response():
  """When ref points to a normalised tool response dict with no explicit path,
  it should unwrap to the original result value."""
  completed = {
    "calc": FakeStepResult({"result": 56.0, "has_more": False})
  }
  r = ref("calc")
  resolved = _resolve_refs(r, completed)
  assert resolved == 56.0


def test_resolve_refs_with_explicit_path_no_unwrap():
  """When ref has explicit path, it should NOT auto-unwrap."""
  completed = {
    "calc": FakeStepResult({"result": 56.0, "has_more": False})
  }
  r = ref("calc", "result")
  resolved = _resolve_refs(r, completed)
  assert resolved == 56.0


def test_resolve_refs_nested_refs():
  """Nested refs in dicts and lists should be resolved."""
  completed = {
    "a": FakeStepResult({"result": 1, "has_more": False}),
    "b": FakeStepResult({"result": 2, "has_more": False}),
  }
  args = {
    "items": [ref("a"), ref("b")],
    "total": ref("a"),
  }
  resolved = _resolve_refs(args, completed)
  assert resolved["items"] == [1, 2]
  assert resolved["total"] == 1


def test_resolve_refs_does_not_unwrap_non_standard_dict():
  """Dicts without 'has_more' should not be unwrapped."""
  completed = {
    "step": FakeStepResult({"custom": "value", "result": "data"})
  }
  r = ref("step")
  resolved = _resolve_refs(r, completed)
  # No has_more field, so should return the raw dict
  assert resolved == {"custom": "value", "result": "data"}


# --------------------------------------------------------------------------- #
# Ref serialization
# --------------------------------------------------------------------------- #

def test_ref_json_serialization():
  """Ref should serialize to JSON via Pydantic."""
  r = ref("step1", "data.value")
  json_str = r.model_dump_json()
  assert '"step_id":"step1"' in json_str
  assert '"path":"data.value"' in json_str


def test_ref_deserialization():
  """Ref should deserialize from JSON."""
  r = ref("step1", "data.value")
  json_str = r.model_dump_json()
  restored = Ref.model_validate_json(json_str)
  assert restored.step_id == "step1"
  assert restored.path == "data.value"


def test_ref_in_dict_serialization():
  """Ref inside a dict should be serializable."""
  args = {"x": ref("a"), "y": ref("b", "result")}
  # Use Pydantic to serialize the dict (simulating Step.args)
  from pydantic import TypeAdapter
  ta = TypeAdapter(dict)
  json_str = ta.dump_json(args)
  assert b"__type__" not in json_str  # BaseModel handles its own serialization


# --------------------------------------------------------------------------- #
# Ref in Plan/Step context
# --------------------------------------------------------------------------- #

def test_step_args_with_ref():
  """Step can be constructed with Ref in args."""
  from nonoka.core.plan import Step
  step = Step(
    id="s2",
    tool="add",
    args={"a": ref("s1"), "b": 2},
  )
  assert isinstance(step.args["a"], Ref)
  assert step.args["a"].step_id == "s1"


def test_plan_builder_auto_detects_ref_dependencies():
  """PlanBuilder should auto-detect dependencies from Ref values."""
  from nonoka.core.plan import PlanBuilder
  plan = (
    PlanBuilder()
    .step("s1", "fetch", url="http://example.com")
    .step("s2", "process", data=ref("s1"))
    .build()
  )
  s2 = plan.get_step("s2")
  assert "s1" in s2.depends_on
