from __future__ import annotations

from nonoka.ext.eval.harbor import (
  _TERMINAL_SYSTEM_PROMPT,
  _terminal_observation,
  _terminal_result,
  atif_from_trace,
  trace_metrics,
)


def _trace() -> dict:
  return {
    "schema_version": 2,
    "started_at": "2026-07-22T00:00:00+00:00",
    "turns": [
      {
        "requested_at": "2026-07-22T00:00:01+00:00",
        "response": {
          "content": "inspect", "usage": {"prompt_tokens": 12, "completion_tokens": 4},
          "tool_calls": [{"id": "call-1"}],
        },
      },
      {
        "requested_at": "2026-07-22T00:00:02+00:00",
        "response": {
          "content": "edit", "usage": {"prompt_tokens": 15, "completion_tokens": 3},
          "tool_calls": [{"id": "call-2"}],
        },
      },
    ],
    "tool_calls": [
      {"id": "call-1", "name": "execute_terminal", "arguments": {"command": "pwd"}, "result": "{}"},
      {"id": "call-2", "name": "execute_terminal", "arguments": {"command": "touch answer"}, "result": "{}"},
    ],
    "extensions": [{"name": "workspace_progress", "phase": "after_tool_batch"}],
    "termination": {"success": True},
  }


def test_harbor_atif_preserves_trace_usage_and_tool_observations():
  trace = _trace()

  assert trace_metrics(trace) == {
    "llm_calls": 2, "tool_calls": 2, "input_tokens": 27, "output_tokens": 7,
  }
  atif = atif_from_trace(trace, session_id="session-1", model="test-model", instruction="solve it")

  assert atif["schema_version"] == "ATIF-v1.7"
  assert atif["session_id"] == "session-1"
  assert atif["final_metrics"] == {
    "total_prompt_tokens": 27, "total_completion_tokens": 7, "total_steps": 3,
  }
  assert atif["steps"][1]["tool_calls"][0]["function_name"] == "execute_terminal"
  assert atif["steps"][1]["observation"]["results"][0]["content"] == "{}"
  assert atif["steps"][2]["tool_calls"][0]["arguments"] == {"command": "touch answer"}
  assert atif["extra"]["extensions"][0]["name"] == "workspace_progress"
  assert atif["extra"]["termination"] == {"success": True}


def test_terminal_result_handles_environment_result_shapes():
  assert _terminal_result("ok") == "ok"
  assert '"stdout": "ok"' in _terminal_result({"stdout": "ok"})

  class Result:
    stdout = "ok"
    stderr = ""
    exit_code = 0

  rendered = _terminal_result(Result())
  assert '"stdout": "ok"' in rendered
  assert '"exit_code": 0' in rendered


def test_terminal_observation_bounds_large_output_and_keeps_exit_status():
  response = _terminal_observation({"stdout": "x" * 100, "stderr": "err", "return_code": 0}, max_chars=30)
  value = response.to_dict()

  assert value["result"]["return_code"] == 0
  assert value["result"]["stderr"] == "err"
  assert "truncated" in value["result"]["stdout"]
  assert value["metadata"]["truncated"] is True
  assert value["metadata"]["omitted_chars"] == 70


def test_terminal_prompt_requires_exact_user_supplied_replacements():
  assert "byte-for-byte" in _TERMINAL_SYSTEM_PROMPT
  assert "inventing a generic substitute" in _TERMINAL_SYSTEM_PROMPT
