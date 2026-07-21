from __future__ import annotations

import json

from nonoka import Runner
from nonoka.core.llm import LLMResponse
from nonoka.ext.eval.datasets.builtins import load_tool_use
from nonoka.ext.eval.runners.headless import HeadlessEvalRunner


class ScriptedProvider:
  def __init__(self, responses):
    self.responses = list(responses)
    self.calls = []

  async def chat(self, **kwargs):
    self.calls.append(kwargs)
    return self.responses.pop(0)


def test_headless_runner_verifies_multistep_tool_sample():
  read_call = {
    "id": "read-1",
    "function": {"name": "read_file", "arguments": json.dumps({"path": "incoming.txt"})},
  }
  tool_call = {
    "id": "write-1",
    "function": {
      "name": "write_file",
      "arguments": json.dumps({
        "path": "cleaned.txt", "content": "done one\ndone two\ndone three\n",
      }),
    },
  }
  verify_call = {
    "id": "verify-1",
    "function": {
      "name": "execute_python",
      "arguments": json.dumps({
        "code": "from pathlib import Path; assert Path('cleaned.txt').read_text().count('\\n') == 3",
      }),
    },
  }
  provider = ScriptedProvider([
    LLMResponse(tool_calls=[read_call], usage={"prompt_tokens": 10, "completion_tokens": 4}),
    LLMResponse(tool_calls=[tool_call], usage={"prompt_tokens": 8, "completion_tokens": 2}),
    LLMResponse(tool_calls=[verify_call], usage={"prompt_tokens": 7, "completion_tokens": 2}),
    LLMResponse(content="verified", usage={"prompt_tokens": 6, "completion_tokens": 1}),
  ])

  def factory(hooks):
    runner = Runner(checkpoint="memory", memory="in_memory", hooks=hooks)
    runner._create_llm = lambda _agent: provider  # type: ignore[method-assign]
    return runner

  result = __import__("asyncio").run(
    HeadlessEvalRunner("fake", runner_factory=factory).evaluate(load_tool_use(1)[0])
  )
  assert result.success, result.verifier_message
  assert result.metrics.llm_calls == 4
  assert result.metrics.tool_calls == 3
  assert result.metrics.total_tokens == 40
  assert result.tool_trace == ["read_file:incoming.txt", "write_file:cleaned.txt", "execute_python"]
  assert {call["temperature"] for call in provider.calls} == {0.0}
