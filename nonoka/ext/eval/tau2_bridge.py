"""One-request JSON bridge from an isolated τ³ harness to current Nonoka.

τ³ pins a LiteLLM version incompatible with Nonoka's runtime.  This process
therefore stays in the Nonoka environment while the official τ³ harness owns
the conversation, tools, user simulator, and reward calculation.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


def _load_environment() -> None:
  for path in (Path.home() / ".config" / "nonoka" / ".env", Path.cwd() / ".env"):
    if path.exists():
      load_dotenv(path, override=False)
  config_path = Path.home() / ".config" / "nonoka" / "config.yaml"
  if config_path.exists():
    import yaml

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not os.environ.get("OPENAI_API_KEY") and config.get("api_key"):
      os.environ["OPENAI_API_KEY"] = str(config["api_key"])
    if not os.environ.get("OPENAI_BASE_URL") and config.get("base_url"):
      os.environ["OPENAI_BASE_URL"] = str(config["base_url"])
  # httpx used by LiteLLM does not accept a SOCKS URL without an optional
  # extra. Keep the user's HTTP(S) proxy intact for this child process.
  for key in ("ALL_PROXY", "all_proxy"):
    if os.environ.get(key, "").lower().startswith("socks://"):
      os.environ.pop(key, None)


class _Bridge:
  """Reusable provider cache for a line-delimited bridge session."""

  def __init__(self) -> None:
    self._providers: dict[str, Any] = {}

  async def respond(self, payload: dict[str, Any]) -> dict[str, Any]:
    from nonoka import Agent, Runner
    from nonoka.core.llm import LLMMessage

    model = str(payload["model"])
    provider = self._providers.get(model)
    if provider is None:
      runner = Runner(checkpoint="memory", memory="in_memory")
      provider = runner._create_llm(Agent(model=model))
      self._providers[model] = provider
    messages = [LLMMessage.model_validate(message) for message in payload["messages"]]
    response = await provider.chat(messages, tools=payload.get("tools"), temperature=0)
    return response.model_dump()


def _emit(payload: dict[str, Any]) -> None:
  print(json.dumps(payload), flush=True)


def _serve() -> int:
  _load_environment()
  bridge = _Bridge()
  for line in sys.stdin:
    if not line.strip():
      continue
    try:
      payload = json.loads(line)
      response = asyncio.run(bridge.respond(payload))
    except Exception as exc:
      _emit({"error": f"{type(exc).__name__}: {exc}"})
      continue
    _emit(response)
  return 0


async def _respond_once(payload: dict[str, Any]) -> dict[str, Any]:
  """Compatibility helper for unit tests and one-shot stdin callers."""
  _load_environment()
  return await _Bridge().respond(payload)


def main() -> int:
  return _serve()


if __name__ == "__main__":
  raise SystemExit(main())
