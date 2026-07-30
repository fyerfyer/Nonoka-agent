from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from nonoka import Agent, ExternalCapability, Runner, tool
from nonoka.core.extensions import ExtensionDecision, LoopExtensionContext, LoopExtensionManager
from nonoka.core.llm import LLMResponse, LLMStreamChunk
from nonoka.core.paradigm import EvaluationResult
from nonoka.core.runtime import (
  CompleteObservationRule, CompletionContract, RuntimeUsage, WorkspaceMutationRule,
)
from nonoka.ext.coding import (
  CodeStrategy, CodeStrategyRouter, CodingWorkflow, ResponseGroundingExtension,
  TerminalCodingWorkflow, TerminalCommandEvaluator, VerifierDiagnosticCode, VerifierRepairExtension,
  WorkspaceProgressExtension,
)


class ScriptedProvider:
  def __init__(self, responses):
    self.responses = list(responses)

  async def chat(self, **_kwargs):
    return self.responses.pop(0)


def make_runner(responses):
  provider = ScriptedProvider(responses)
  runner = Runner(checkpoint="memory", memory="in_memory")
  runner._create_llm = lambda _agent: provider  # type: ignore[method-assign]
  return runner


@pytest.mark.asyncio
async def test_verifier_extension_repairs_in_one_react_session():
  class Verifier:
    def __init__(self):
      self.calls = 0

    async def evaluate(self, result):
      self.calls += 1
      return EvaluationResult(
        passed=result.data == "fixed answer",
        feedback="The answer must be fixed.",
      )

  verifier = Verifier()
  runner = make_runner([LLMResponse(content="first answer"), LLMResponse(content="fixed answer")])
  agent = Agent(model="fake", extensions=[VerifierRepairExtension(verifier, max_repairs=1)], max_turns=3)

  result = await runner.run_react(agent, "Return an answer", deps=None)

  assert result.success is True
  assert result.data == "fixed answer"
  assert verifier.calls == 2
  assert any("Verifier feedback" in entry.content for entry in result.session.memory.entries)
  assert len(result.trace["verifications"]) == 2
  assert any(entry["name"] == "verifier_repair" for entry in result.trace["extensions"])


@pytest.mark.asyncio
async def test_verifier_extension_fails_when_repair_budget_is_exhausted():
  class RejectingVerifier:
    async def evaluate(self, _result):
      return EvaluationResult(passed=False, feedback="still invalid")

  runner = make_runner([LLMResponse(content="bad")])
  agent = Agent(model="fake", extensions=[VerifierRepairExtension(RejectingVerifier(), max_repairs=0)])

  result = await runner.run_react(agent, "Return an answer", deps=None)

  assert result.success is False
  assert result.error_type == "extension_rejected"
  assert "still invalid" in result.error


@pytest.mark.asyncio
async def test_grounding_extension_revises_unverified_final_claim():
  def validator(_context, content):
    return True if "10 available" in content else "State evidence says there are 10 available options."

  runner = make_runner([
    LLMResponse(content="There are 12 options."),
    LLMResponse(content="There are 10 available options."),
  ])
  agent = Agent(model="fake", extensions=[ResponseGroundingExtension(validator, max_repairs=1)])

  result = await runner.run_react(agent, "Tell the customer the available count", deps=None)

  assert result.success is True
  assert result.data == "There are 10 available options."
  assert any("Grounding feedback" in entry.content for entry in result.session.memory.entries)


@pytest.mark.asyncio
async def test_after_tool_batch_extension_can_add_guidance_without_altering_tool_execution():
  class ToolGuidance:
    name = "tool_guidance"

    async def after_tool_batch(self, context: LoopExtensionContext):
      assert context.tool_results == [{"result": "written", "has_more": False}]
      return ExtensionDecision(feedback="Now summarize the confirmed tool result.")

  @tool
  async def write_value(ctx, value: str):
    return "written"

  tool_call = {
    "id": "write-1",
    "function": {"name": "write_value", "arguments": json.dumps({"value": "x"})},
  }
  runner = make_runner([LLMResponse(tool_calls=[tool_call]), LLMResponse(content="confirmed")])
  agent = Agent(model="fake", tools=[write_value], extensions=[ToolGuidance()], max_turns=3)

  result = await runner.run_react(agent, "Write a value", deps=None)

  assert result.success is True
  assert any("Now summarize" in entry.content for entry in result.session.memory.entries)
  assert result.trace["extensions"][-1]["name"] == "tool_guidance"


@pytest.mark.asyncio
async def test_extension_can_restrict_a_turn_to_final_response_only():
  observed_tools = []

  class FinalizeOnly:
    name = "finalize_only"

    async def before_turn(self, _context):
      return ExtensionDecision(feedback="Finish now.", disable_tools=True)

  class Provider:
    async def chat(self, **kwargs):
      observed_tools.append(kwargs.get("tools"))
      return LLMResponse(content="finished")

  runner = Runner(checkpoint="memory", memory="in_memory")
  runner._create_llm = lambda _agent: Provider()  # type: ignore[method-assign]
  agent = Agent(model="fake", tools=[], extensions=[FinalizeOnly()])

  result = await runner.run_react(agent, "finish", deps=None)

  assert result.success is True
  assert observed_tools == [None]
  assert any("Finish now" in entry.content for entry in result.session.memory.entries)


@pytest.mark.asyncio
async def test_workspace_progress_disables_tools_only_after_contract_is_satisfied():
  extension = WorkspaceProgressExtension()
  usage = RuntimeUsage(mutation_count=1)
  session = SimpleNamespace(
    completion_contract=CompletionContract(rules=(WorkspaceMutationRule(),)),
    runtime_state=SimpleNamespace(usage=usage),
    extension_state={},
  )

  decision = await extension.before_turn(
    LoopExtensionContext(session=session, runner=None, prompt="", turn=3)
  )

  assert decision.disable_tools is True
  assert decision.details["phase"] == "finalization"

  usage.mutation_count = 0
  decision = await extension.before_turn(
    LoopExtensionContext(session=session, runner=None, prompt="", turn=4)
  )
  assert decision.disable_tools is False


@pytest.mark.asyncio
async def test_workspace_progress_waits_for_explicit_focused_marker():
  extension = WorkspaceProgressExtension()
  usage = RuntimeUsage(
    mutation_count=1,
    focused_verification_status="passed",
    last_passed_focused_at_observation=2,
    last_effect_at_observation=1,
    latest_verification={"command": "pytest -q tests/test_target.py"},
  )
  session = SimpleNamespace(
    completion_contract=CompletionContract(
      require_workspace_mutation=True,
      require_focused_verification=True,
    ),
    runtime_state=SimpleNamespace(usage=usage),
    extension_state={},
  )

  ordinary = await extension.before_turn(
    LoopExtensionContext(session=session, runner=None, prompt="", turn=3)
  )
  usage.latest_verification = {
    "command": "NONOKA_VERIFY=focused pytest -q tests/test_target.py"
  }
  explicit = await extension.before_turn(
    LoopExtensionContext(session=session, runner=None, prompt="", turn=4)
  )

  assert ordinary.disable_tools is False
  assert ordinary.details["awaiting_explicit_focused_check"] is True
  assert explicit.disable_tools is True


@pytest.mark.asyncio
async def test_tool_call_during_finalization_is_rejected_without_execution():
  calls = 0

  @tool
  async def forbidden_tool(ctx):
    nonlocal calls
    calls += 1
    return "should not run"

  class FinalizeOnly:
    name = "finalize_only"

    async def before_turn(self, _context):
      return ExtensionDecision(disable_tools=True)

  tool_call = {
    "id": "forbidden-1",
    "function": {"name": "forbidden_tool", "arguments": "{}"},
  }
  runner = make_runner([
    LLMResponse(tool_calls=[tool_call]),
    LLMResponse(tool_calls=[tool_call]),
  ])
  agent = Agent(
    model="fake",
    tools=[forbidden_tool],
    extensions=[FinalizeOnly()],
    max_turns=2,
  )

  result = await runner.run_react(agent, "finish", deps=None)

  assert result.success is False
  assert result.error_type == "extension_rejected"
  assert calls == 0


@pytest.mark.asyncio
async def test_streaming_finalization_suppresses_hallucinated_host_tool_and_recovers():
  observed_tools = []

  class FinalizeOnly:
    name = "finalize_only"

    async def before_turn(self, _context):
      return ExtensionDecision(disable_tools=True)

  class Provider:
    calls = 0

    async def chat_stream(self, **kwargs):
      observed_tools.append(kwargs.get("tools"))
      self.calls += 1
      if self.calls == 1:
        yield LLMStreamChunk(tool_call_deltas=[{
          "index": 0,
          "id": "host-1",
          "function": {"name": "host_tool", "arguments": "{}"},
        }])
        yield LLMStreamChunk(finish_reason="tool_calls")
      else:
        yield LLMStreamChunk(content_delta="finished")
        yield LLMStreamChunk(finish_reason="stop")

  runner = Runner(checkpoint="memory", memory="in_memory")
  runner._create_llm = lambda _agent: Provider()  # type: ignore[method-assign]
  agent = Agent(
    model="fake",
    tools=[ExternalCapability(
      name="host_tool", description="host", parameters={"type": "object"},
    )],
    extensions=[FinalizeOnly()],
    max_turns=3,
  )

  events = [event async for event in runner.run_react_stream(agent, "finish", deps=None)]

  assert events[-1].type == "final"
  assert events[-1].data["data"] == "finished"
  assert not any(event.type.startswith("tool_call") for event in events)
  assert observed_tools == [None, None]


def test_extension_names_must_be_unique():
  class Extension:
    name = "same"

  with pytest.raises(ValueError, match="names must be unique"):
    LoopExtensionManager([Extension(), Extension()])


def test_code_strategy_router_requires_explicit_verifier_capability():
  router = CodeStrategyRouter()

  assert router.choose(deterministic_verifier=False, requires_workspace=False) is CodeStrategy.DIRECT
  assert router.choose(deterministic_verifier=False, requires_workspace=True) is CodeStrategy.TOOL_ASSISTED
  assert router.choose(deterministic_verifier=True, requires_workspace=True) is CodeStrategy.VERIFIED_REPAIR
  with pytest.raises(ValueError, match="requires a deterministic evaluator"):
    router.extensions_for(CodeStrategy.VERIFIED_REPAIR, evaluator=None)


def test_coding_workflow_defaults_to_direct_without_workspace():
  workflow = CodingWorkflow()
  agent = workflow.build_agent(model="fake", tools=[])

  assert workflow.resolve_strategy() is CodeStrategy.DIRECT
  assert list(agent.tools) == []


def test_terminal_workflow_requires_explicit_verify_command_for_repair():
  class Evaluator:
    async def evaluate(self, _result):
      return EvaluationResult(passed=True)

  workflow = TerminalCodingWorkflow(
    requires_workspace=True, evaluator=Evaluator(), strategy=CodeStrategy.VERIFIED_REPAIR,
  )
  with pytest.raises(ValueError, match="explicit verify_command"):
    workflow.resolve_strategy()


@pytest.mark.asyncio
async def test_terminal_command_evaluator_uses_only_explicit_command_and_reports_failure():
  commands: list[tuple[str, ...]] = []

  async def execute(command: tuple[str, ...]):
    commands.append(command)
    return {"exit_code": 1, "stdout": "", "stderr": "AssertionError: expected 2"}

  evaluator = TerminalCommandEvaluator(("pytest", "-q", "tests/test_target.py"), execute)
  report = await evaluator.evaluate(None)

  assert commands == [("pytest", "-q", "tests/test_target.py")]
  assert report.passed is False
  assert report.diagnostic is not None
  assert report.diagnostic.code is VerifierDiagnosticCode.TEST_FAILURE
  assert "AssertionError" in report.message


@pytest.mark.asyncio
async def test_workspace_progress_extension_reminds_only_after_exploration_budget():
  extension = WorkspaceProgressExtension(max_exploration_turns=2)
  session = type("Session", (), {})()
  context = LoopExtensionContext(
    session=session, runner=None, prompt="edit the workspace", turn=1,
    tool_calls=[{"function": {"arguments": '{"command":"find . -type f"}'}}],
  )

  first = await extension.after_tool_batch(context)
  context.turn = 2
  second = await extension.after_tool_batch(context)
  context.turn = 3
  context.tool_calls = [{"function": {"arguments": '{"command":"sed -i s/old/new/ file"}'}}]
  third = await extension.after_tool_batch(context)

  assert first.feedback is None
  assert "workspace change" in (second.feedback or "")
  assert third.details["mutation_command_seen"] is True


@pytest.mark.asyncio
async def test_workspace_progress_extension_does_not_treat_stderr_redirect_as_mutation():
  extension = WorkspaceProgressExtension(max_exploration_turns=1)
  session = type("Session", (), {})()
  context = LoopExtensionContext(
    session=session, runner=None, prompt="edit", turn=1,
    tool_calls=[{"function": {"arguments": '{"command":"find / -name target 2>/dev/null"}'}}],
  )

  decision = await extension.after_tool_batch(context)

  assert decision.details["mutation_command_seen"] is False
  assert "workspace change" in (decision.feedback or "")


@pytest.mark.asyncio
async def test_workspace_progress_extension_uses_persisted_session_state():
  extension = WorkspaceProgressExtension(max_exploration_turns=2)
  session = type("Session", (), {"extension_state": {}})()
  context = LoopExtensionContext(
    session=session, runner=None, prompt="edit", turn=1,
    tool_calls=[{"function": {"arguments": '{"command":"find . -type f"}'}}],
  )

  first = await extension.after_tool_batch(context)
  restored = type(
    "Session", (), {"extension_state": json.loads(json.dumps(session.extension_state))}
  )()
  context.session = restored
  context.turn = 2
  second = await extension.after_tool_batch(context)

  assert first.feedback is None
  assert "workspace change" in (second.feedback or "")
  assert restored.extension_state["workspace_progress"]["exploration_turns"] == 2


@pytest.mark.asyncio
async def test_workspace_progress_extension_requires_post_effect_success_before_completion_nudge():
  extension = WorkspaceProgressExtension(max_post_verification_batches=2)
  usage = SimpleNamespace(
    effect_count=1,
    successful_command_count=1,
    last_effect_at_observation=1,
    last_successful_command_at_observation=1,
    last_partial_observation_at=None,
    last_complete_observation_at=1,
  )
  session = SimpleNamespace(
    extension_state={},
    runtime_state=SimpleNamespace(usage=usage),
  )
  context = LoopExtensionContext(
    session=session, runner=None, prompt="configure", turn=1,
    tool_calls=[{"function": {"arguments": '{"command":"apply-change"}'}}],
  )

  same_batch = await extension.after_tool_batch(context)
  assert same_batch.feedback is None

  usage.successful_command_count = 2
  usage.last_successful_command_at_observation = 2
  context.turn = 2
  context.tool_calls = [{"function": {"arguments": '{"command":"check-change"}'}}]
  verified = await extension.after_tool_batch(context)

  assert "Completion evidence" in (verified.feedback or "")
  assert verified.details["phase"] == "verification_ready"


@pytest.mark.asyncio
async def test_workspace_progress_extension_accepts_complete_post_effect_observation_without_exit_code():
  extension = WorkspaceProgressExtension(max_post_verification_batches=2)
  usage = SimpleNamespace(
    effect_count=1,
    successful_command_count=0,
    last_effect_at_observation=4,
    last_successful_command_at_observation=None,
    last_partial_observation_at=None,
    last_complete_observation_at=5,
  )
  session = SimpleNamespace(
    extension_state={}, runtime_state=SimpleNamespace(usage=usage),
  )
  context = LoopExtensionContext(
    session=session, runner=None, prompt="", turn=2,
    tool_calls=[], tool_results=[],
  )

  decision = await extension.after_tool_batch(context)

  assert "Completion evidence" in (decision.feedback or "")
  assert decision.details["post_effect_complete"] is True
  assert decision.details["successful_command_count"] == 0


@pytest.mark.asyncio
async def test_workspace_progress_requests_bounded_follow_up_immediately_after_partial_observation():
  extension = WorkspaceProgressExtension()
  usage = RuntimeUsage(
    observation_count=4,
    partial_observation_count=1,
    last_partial_observation_at=4,
    last_complete_observation_at=3,
    latest_partial_tool="read",
    latest_partial_artifact_ref="artifacts/read-4.txt",
  )
  session = SimpleNamespace(
    completion_contract=CompletionContract(rules=(CompleteObservationRule(),)),
    extension_state={},
    runtime_state=SimpleNamespace(usage=usage),
  )
  context = LoopExtensionContext(
    session=session, runner=None, prompt="", turn=2, tool_calls=[], tool_results=[],
  )

  decision = await extension.after_tool_batch(context)

  assert decision.details["phase"] == "observation_recovery"
  assert "bounded artifact" in (decision.feedback or "")
  assert usage.observation_feedback_after == 4

  usage.observation_count = 5
  usage.last_complete_observation_at = 5
  follow_up = await extension.after_tool_batch(context)

  assert follow_up.details["phase"] == "exploration"
  assert session.completion_contract.unmet_requirements(usage) == []


@pytest.mark.asyncio
async def test_workspace_progress_extension_has_bounded_post_verification_reminder():
  extension = WorkspaceProgressExtension(max_post_verification_batches=2)
  usage = SimpleNamespace(
    effect_count=1,
    successful_command_count=2,
    last_effect_at_observation=1,
    last_successful_command_at_observation=2,
    last_partial_observation_at=None,
    last_complete_observation_at=2,
  )
  session = SimpleNamespace(
    extension_state={
      "workspace_progress": {
        "exploration_turns": 0,
        "mutating": True,
        "effect_count": 1,
        "successful_command_count": 2,
        "verification_ready": True,
        "post_verification_batches": 0,
        "completion_reminded": True,
      }
    },
    runtime_state=SimpleNamespace(usage=usage),
  )
  context = LoopExtensionContext(
    session=session, runner=None, prompt="configure", turn=3,
    tool_calls=[{"function": {"arguments": '{"command":"inspect-again"}'}}],
  )

  first = await extension.after_tool_batch(context)
  context.turn = 4
  second = await extension.after_tool_batch(context)

  assert first.feedback is None
  assert "Verification budget" in (second.feedback or "")
  assert second.details["verification_budget_reached"] is True
