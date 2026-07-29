from __future__ import annotations

from nonoka.core.agent import Agent
from nonoka.core.runtime import (
  CommandSucceededRule,
  CompleteObservationRule,
  CompletionContract,
  ObservedEffectRule,
  PathsChangedRule,
  RuntimeUsage,
  VerificationPassedRule,
  WorkspaceMutationRule,
)
from nonoka.core.session import Session


def test_completion_contract_evaluates_declarative_rules():
  contract = CompletionContract(rules=(
    WorkspaceMutationRule(),
    PathsChangedRule(paths=("src/app.py",)),
    CommandSucceededRule(command="pytest -q"),
  ))

  assert contract.unmet_requirements(RuntimeUsage()) == [
    "make at least 1 observed workspace mutation(s)",
    "modify the required paths: src/app.py",
    "run the verification command successfully: pytest -q",
  ]
  assert contract.unmet_requirements(RuntimeUsage(
    mutation_count=1,
    changed_paths=["src/app.py"],
    verified_commands=["pytest -q"],
  )) == []


def test_completion_contract_compiles_legacy_fields_into_rules():
  contract = CompletionContract(
    require_workspace_mutation=True,
    required_paths=("answer.txt",),
    verification_command="test -s answer.txt",
  )

  kinds = [rule.kind for rule in contract.effective_rules()]
  assert kinds == ["workspace_mutation", "paths_changed", "command_succeeded"]


def test_observed_effect_is_distinct_from_workspace_mutation():
  contract = CompletionContract(require_observed_effect=True)

  assert contract.unmet_requirements(RuntimeUsage(mutation_count=3)) == [
    "produce at least 1 host-observed task effect(s)"
  ]
  assert contract.unmet_requirements(RuntimeUsage(effect_count=1)) == []
  assert isinstance(contract.effective_rules()[0], ObservedEffectRule)


def test_completion_contract_rules_round_trip_through_json():
  contract = CompletionContract(rules=(
    PathsChangedRule(paths=("a", "b")),
    CommandSucceededRule(command="check"),
    CompleteObservationRule(),
    VerificationPassedRule(),
  ))

  restored = CompletionContract.model_validate_json(contract.model_dump_json())

  assert isinstance(restored.rules[0], PathsChangedRule)
  assert isinstance(restored.rules[1], CommandSucceededRule)
  assert isinstance(restored.rules[2], CompleteObservationRule)
  assert isinstance(restored.rules[3], VerificationPassedRule)


def test_focused_verification_must_follow_latest_effect():
  contract = CompletionContract(require_focused_verification=True)
  usage = RuntimeUsage(
    focused_verification_status="passed",
    last_passed_focused_at_observation=2,
    last_effect_at_observation=2,
  )

  assert contract.unmet_requirements(usage) == [
    "run and pass a fresh focused verification check after the latest change"
  ]

  usage.last_passed_focused_at_observation = 3
  assert contract.unmet_requirements(usage) == []


def test_focused_verification_reports_failed_and_unavailable_states():
  contract = CompletionContract(require_focused_verification=True)

  assert "repair" in contract.unmet_requirements(RuntimeUsage(
    focused_verification_status="failed"
  ))[0]
  assert "available" in contract.unmet_requirements(RuntimeUsage(
    focused_verification_status="unavailable"
  ))[0]


def test_complete_observation_rule_requires_fresh_complete_evidence():
  contract = CompletionContract(require_complete_observations=True)
  usage = RuntimeUsage(
    observation_count=1,
    partial_observation_count=1,
    last_partial_observation_at=1,
  )

  assert contract.unmet_requirements(usage)

  usage.observation_count = 2
  usage.last_complete_observation_at = 2
  assert contract.unmet_requirements(usage), "pre-feedback observations must not clear the rule"

  usage.observation_feedback_after = 2
  usage.observation_count = 3
  usage.last_complete_observation_at = 3

  assert contract.unmet_requirements(usage) == []

  usage.observation_count = 4
  usage.last_partial_observation_at = 4
  assert contract.unmet_requirements(usage)

  usage.observation_count = 5
  usage.last_complete_observation_at = 5
  assert contract.unmet_requirements(usage) == []


def test_advisory_completion_contract_preserves_scorable_state_after_correction():
  agent = Agent(
    model="gpt-4o",
    completion_contract=CompletionContract(
      require_complete_observations=True,
      max_corrections=0,
      enforcement="advisory",
    ),
  )
  session = Session("advisory-contract", agent)
  session.runtime_state.usage.last_partial_observation_at = 1

  assert session.completion_feedback() is None
  assert session.runtime_state.usage.completion_warning_count == 1
  assert "complete follow-up observation" in (session.runtime_state.usage.latest_completion_warning or "")
