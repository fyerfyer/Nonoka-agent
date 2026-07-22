"""Run a Nonoka-backed agent through the official τ³-bench harness.

Execute this file with τ³'s Python 3.12 environment, not with Nonoka's
environment.  Set ``NONOKA_TAU_BRIDGE_PYTHON`` to the Python executable that
contains the local nonoka-agent editable installation.
"""

from __future__ import annotations

import json
import os
import select
import subprocess
import uuid
from typing import Any


class _NonokaBridgeClient:
  """Persistent NDJSON client for the isolated Nonoka runtime."""

  def __init__(self, bridge_python: str) -> None:
    self.process = subprocess.Popen(
      [bridge_python, "-m", "nonoka.ext.eval.tau2_bridge"],
      stdin=subprocess.PIPE,
      stdout=subprocess.PIPE,
      stderr=subprocess.DEVNULL,
      text=True,
      bufsize=1,
    )

  def request(self, payload: dict[str, Any]) -> dict[str, Any]:
    if self.process.poll() is not None or self.process.stdin is None or self.process.stdout is None:
      raise RuntimeError("Nonoka bridge exited before responding.")
    self.process.stdin.write(json.dumps(payload) + "\n")
    self.process.stdin.flush()
    ready, _, _ = select.select([self.process.stdout], [], [], 120)
    if not ready:
      self.close()
      raise TimeoutError("Nonoka bridge timed out after 120 seconds.")
    line = self.process.stdout.readline()
    try:
      response = json.loads(line)
    except json.JSONDecodeError as exc:
      raise RuntimeError("Nonoka bridge returned invalid NDJSON.") from exc
    if response.get("error"):
      raise RuntimeError(str(response["error"]))
    return response

  def close(self) -> None:
    if self.process.poll() is not None:
      return
    if self.process.stdin is not None:
      self.process.stdin.close()
    try:
      self.process.wait(timeout=3)
    except subprocess.TimeoutExpired:
      self.process.terminate()
      try:
        self.process.wait(timeout=3)
      except subprocess.TimeoutExpired:
        self.process.kill()
        self.process.wait(timeout=3)


_bridge_client: _NonokaBridgeClient | None = None


def _sanitize_proxy_for_tau() -> None:
  """Keep HTTP(S) proxying while hiding unsupported SOCKS ALL_PROXY."""
  for key in ("ALL_PROXY", "all_proxy"):
    if os.environ.get(key, "").lower().startswith("socks://"):
      os.environ.pop(key, None)


def _configure_official_evaluator() -> None:
  """Point τ³'s built-in NL judge at the configured benchmark model.

  The official harness otherwise hard-codes an OpenAI-only GPT default for
  natural-language assertions.  Nonoka intentionally keeps τ³'s evaluator;
  this only supplies the model identifier used by that evaluator so an
  OpenAI-compatible endpoint can score every task in the same run.
  """
  model = os.environ.get("NONOKA_TAU_EVALUATOR_MODEL")
  if not model:
    return
  from tau2.evaluator import evaluator_nl_assertions

  # τ³'s pinned LiteLLM requires an explicit provider for custom OpenAI
  # endpoints, whereas Nonoka accepts the configured model alias directly.
  evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS = (
    model if "/" in model else f"openai/{model}"
  )


def _tau_message_to_nonoka(message: Any) -> dict[str, Any]:
  from tau2.data_model.message import AssistantMessage, SystemMessage, ToolMessage, UserMessage

  if isinstance(message, (SystemMessage, UserMessage)):
    return {"role": message.role, "content": message.content}
  if isinstance(message, AssistantMessage):
    tool_calls = None
    if message.tool_calls:
      tool_calls = [
        {
          "id": call.id,
          "type": "function",
          "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
        }
        for call in message.tool_calls
      ]
    return {"role": "assistant", "content": message.content, "tool_calls": tool_calls}
  if isinstance(message, ToolMessage):
    return {"role": "tool", "content": message.content, "tool_call_id": message.id}
  raise TypeError(f"Unsupported τ³ message: {type(message).__name__}")


def _to_tau_tool_calls(raw_calls: list[dict[str, Any]] | None) -> list[Any]:
  from tau2.data_model.message import ToolCall

  calls = []
  for raw in raw_calls or []:
    function = raw.get("function", raw)
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
      try:
        arguments = json.loads(arguments)
      except json.JSONDecodeError:
        arguments = {}
    calls.append(ToolCall(
      id=str(raw.get("id") or uuid.uuid4().hex),
      name=str(function["name"]), arguments=arguments, requestor="assistant",
    ))
  return calls


def _ask_nonoka(model: str, messages: list[Any], tools: list[Any]) -> dict[str, Any]:
  global _bridge_client
  bridge_python = os.environ.get("NONOKA_TAU_BRIDGE_PYTHON")
  if not bridge_python:
    raise RuntimeError("Set NONOKA_TAU_BRIDGE_PYTHON to the isolated Nonoka Python executable.")
  request = {
    "model": model,
    "messages": [_tau_message_to_nonoka(message) for message in messages],
    "tools": [tool.openai_schema for tool in tools],
  }
  if _bridge_client is None:
    _bridge_client = _NonokaBridgeClient(bridge_python)
  return _bridge_client.request(request)


def _register_components() -> None:
  """Register the Nonoka agent *and* user simulator with τ³.

  τ³ drives a conversation with two LLM participants.  Routing only the
  evaluated agent through Nonoka is insufficient for an OpenAI-compatible
  endpoint configured in Nonoka: τ³'s pinned LiteLLM would still send the
  simulated-user request to its default endpoint.  The user implementation
  below preserves τ³'s official prompt, scenario, tools and stop semantics,
  while delegating only its model completion to the same isolated Nonoka
  bridge as the evaluated agent.
  """
  from pydantic import BaseModel
  from tau2.agent.base_agent import HalfDuplexAgent, ValidAgentInputMessage
  from tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
  )
  from tau2.registry import registry
  from tau2.user.user_simulator import UserSimulator
  from tau2.user.user_simulator_base import UserState, ValidUserInputMessage

  class State(BaseModel):
    messages: list[Any]

  class NonokaTauAgent(HalfDuplexAgent[State]):
    def __init__(self, tools, domain_policy, llm: str, **_kwargs):
      super().__init__(tools=tools, domain_policy=domain_policy)
      self.llm = llm

    def get_init_state(self, message_history: list[Message] | None = None) -> State:
      system = SystemMessage(
        role="system",
        content=(
          "You are a customer-service agent. Follow this policy exactly. "
          "On each turn, either send text or make tool calls, never both.\n\n"
          f"<policy>\n{self.domain_policy}\n</policy>"
        ),
      )
      return State(messages=[system, *(message_history or [])])

    def generate_next_message(self, message: ValidAgentInputMessage, state: State):
      if isinstance(message, MultiToolMessage):
        state.messages.extend(message.tool_messages)
      else:
        state.messages.append(message)
      response = _ask_nonoka(self.llm, state.messages, self.tools)
      tool_calls = _to_tau_tool_calls(response.get("tool_calls"))
      assistant = AssistantMessage.text(
        "" if tool_calls else str(response.get("content") or "I need more information."),
        tool_calls=tool_calls or None,
        usage=response.get("usage"), raw_data=response,
      )
      state.messages.append(assistant)
      return assistant, state

  def create_nonoka_tau_agent(tools, domain_policy, **kwargs):
    return NonokaTauAgent(tools, domain_policy, llm=kwargs.get("llm", "deepseek-chat"))

  class NonokaTauUser(UserSimulator):
    """Official τ³ user simulator prompt and state, with Nonoka completion."""

    def _generate_next_message(
      self, message: ValidUserInputMessage, state: UserState,
    ) -> UserMessage:
      if isinstance(message, MultiToolMessage):
        state.messages.extend(message.tool_messages)
      elif isinstance(message, ToolMessage):
        state.messages.append(message)
      elif message.has_content() or message.is_tool_call():
        state.messages.append(message)

      response = _ask_nonoka(
        self.llm,
        state.system_messages + state.flip_roles(),
        self.tools or [],
      )
      tool_calls = _to_tau_tool_calls(response.get("tool_calls"))
      user_message = UserMessage(
        role="user",
        content=str(response.get("content") or ""),
        cost=0.0,
        usage=response.get("usage"),
        raw_data=response,
      )
      if tool_calls:
        user_message.tool_calls = [
          ToolCall(
            id=call.id,
            name=call.name,
            arguments=call.arguments,
            requestor="user",
          )
          for call in tool_calls
        ]
      return user_message

  registry.register_agent_factory(create_nonoka_tau_agent, "nonoka_tau")
  registry.register_user(NonokaTauUser, "nonoka_tau_user")


def main() -> int:
  _sanitize_proxy_for_tau()
  _configure_official_evaluator()
  _register_components()
  from tau2.cli import main as tau_main
  try:
    return int(tau_main() or 0)
  finally:
    if _bridge_client is not None:
      _bridge_client.close()


if __name__ == "__main__":
  raise SystemExit(main())
