"""Hard token/cost budget behavior, including cache accounting."""

import pytest

from nonoka.core.agent import Agent
from nonoka.core.errors import RuntimeTerminatedError
from nonoka.core.runtime import RuntimeLimits, TerminalReason
from nonoka.core.session import Session


def test_token_budget_terminates_session_after_actual_provider_usage():
  session = Session(
    session_id="budget-test",
    agent=Agent(model="test", runtime_limits=RuntimeLimits(max_total_tokens=4)),
    deps=None,
  )

  with pytest.raises(RuntimeTerminatedError) as exc_info:
    session.record_model_usage({"prompt_tokens": 3, "completion_tokens": 2})

  assert exc_info.value.termination.reason == TerminalReason.TOKEN_BUDGET_EXHAUSTED
  assert session.runtime_state.usage.total_tokens == 5


def test_cache_hit_records_savings_without_spending_task_budget():
  session = Session(
    session_id="cache-budget-test",
    agent=Agent(model="test", runtime_limits=RuntimeLimits(max_total_tokens=1, max_cost_usd=0.01)),
    deps=None,
  )

  session.record_model_usage(
    {"prompt_tokens": 40, "completion_tokens": 10, "estimated_cost_usd": 0.2},
    cache_hit=True,
  )

  usage = session.runtime_state.usage
  assert usage.cache_hits == 1
  assert usage.cache_saved_tokens == 50
  assert usage.cache_saved_cost_usd == 0.2
  assert usage.total_tokens == 0
  assert usage.cost_usd == 0
