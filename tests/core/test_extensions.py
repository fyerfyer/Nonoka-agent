from __future__ import annotations

import json

import pytest

from nonoka import Agent, Runner, tool
from nonoka.core.extensions import ExtensionDecision, LoopExtensionContext, LoopExtensionManager
from nonoka.core.llm import LLMResponse
from nonoka.core.paradigm import EvaluationResult
from nonoka.ext.coding import CodeStrategy, CodeStrategyRouter, ResponseGroundingExtension, VerifierRepairExtension


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
