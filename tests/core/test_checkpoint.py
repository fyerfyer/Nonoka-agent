import pytest
from nonoka.core.checkpoint import SessionState, SessionStatus, StepResult, StepError
from nonoka.core.plan import Plan, Step


def test_session_state_serialization_fidelity():
  step1 = Step(id="s1", tool="search", args={"q": "AI"})
  step2 = Step(id="s2", tool="write", depends_on=frozenset(["s1"]))
  plan = Plan(steps=(step1, step2), objective="Research AI")

  state = SessionState(
    session_id="session-123",
    status=SessionStatus.RUNNING,
    current_plan=plan,
    completed_steps={"s1": StepResult(data={"status": "ok"})}
  )

  json_str = state.model_dump_json()
  assert "session-123" in json_str

  restored_state = SessionState.model_validate_json(json_str)

  assert restored_state.session_id == "session-123"
  assert restored_state.status == SessionStatus.RUNNING

  restored_plan = restored_state.current_plan
  assert isinstance(restored_plan, Plan), "Plan type lost during deserialization!"
  assert len(restored_plan.steps) == 2

  restored_step2 = restored_plan.get_step("s2")
  assert restored_step2 is not None

  assert isinstance(restored_step2.depends_on, frozenset), "frozenset was not correctly restored!"
  assert "s1" in restored_step2.depends_on
