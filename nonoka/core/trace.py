"""Portable, redacted execution traces for runs and benchmark artifacts."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


_SECRET_KEY = re.compile(r"(?:api[_-]?key|token|password|secret|authorization|cookie)", re.I)
_SECRET_VALUE = re.compile(r"(?:sk-[A-Za-z0-9_-]{12,}|Bearer\s+[A-Za-z0-9._-]+)", re.I)
_SAFE_USAGE_KEYS = {
  "prompt_tokens", "completion_tokens", "input_tokens", "output_tokens", "total_tokens", "max_tokens",
}


def _now() -> str:
  return datetime.now(timezone.utc).isoformat()


def redact(value: Any, *, key: str | None = None, max_chars: int = 262_144) -> Any:
  """Recursively redact common credentials and bound very large payloads."""
  if key and key.lower() not in _SAFE_USAGE_KEYS and _SECRET_KEY.search(key):
    return "[REDACTED]"
  if isinstance(value, dict):
    return {str(k): redact(v, key=str(k), max_chars=max_chars) for k, v in value.items()}
  if isinstance(value, (list, tuple)):
    return [redact(v, max_chars=max_chars) for v in value]
  if isinstance(value, str):
    text = _SECRET_VALUE.sub("[REDACTED]", value)
    if len(text) > max_chars:
      return {
        "truncated": text[:max_chars],
        "sha256": hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest(),
        "original_chars": len(text),
      }
    return text
  return value


@dataclass
class ExecutionTrace:
  """A serialisable, bounded trace attached to every Session."""

  schema_version: int = 1
  started_at: str = field(default_factory=_now)
  generation: dict[str, Any] = field(default_factory=dict)
  turns: list[dict[str, Any]] = field(default_factory=list)
  tool_calls: list[dict[str, Any]] = field(default_factory=list)
  verifications: list[dict[str, Any]] = field(default_factory=list)
  extensions: list[dict[str, Any]] = field(default_factory=list)
  termination: dict[str, Any] = field(default_factory=dict)

  def record_generation(self, **data: Any) -> None:
    self.generation.update(redact(data))

  def record_turn_request(self, turn: int, messages: Any, **options: Any) -> None:
    self.turns.append({
      "turn": turn, "requested_at": _now(), "messages": redact(messages),
      "options": redact(options),
    })

  def record_turn_response(self, turn: int, response: Any) -> None:
    for entry in reversed(self.turns):
      if entry["turn"] == turn:
        entry["responded_at"] = _now()
        entry["response"] = redact(response)
        return
    self.turns.append({"turn": turn, "responded_at": _now(), "response": redact(response)})

  def record_tool_start(self, call_id: str, name: str, arguments: Any, execution: Any) -> int:
    self.tool_calls.append({
      "id": call_id, "name": name, "arguments": redact(arguments),
      "execution": redact(asdict(execution) if hasattr(execution, "__dataclass_fields__") else execution),
      "started_at": _now(),
    })
    return len(self.tool_calls) - 1

  def record_tool_end(self, index: int, result: Any = None, error: Exception | None = None) -> None:
    if not 0 <= index < len(self.tool_calls):
      return
    entry = self.tool_calls[index]
    entry["ended_at"] = _now()
    if error is not None:
      entry["error"] = {"type": type(error).__name__, "message": redact(str(error))}
    else:
      entry["result"] = redact(result)

  def record_external_receipt(self, call_id: str, receipt: Any, *, verified: bool) -> None:
    """Attach a host execution receipt to its original delegated tool call."""
    for entry in reversed(self.tool_calls):
      if entry.get("id") == call_id:
        entry["external_receipt"] = redact(receipt)
        entry["workspace_audit"] = "verified" if verified else "unverified"
        return

  def record_verification(self, **data: Any) -> None:
    self.verifications.append({"at": _now(), **redact(data)})

  def record_extension(self, **data: Any) -> None:
    """Record an extension decision without exposing extension internals."""
    self.extensions.append({"at": _now(), **redact(data)})

  def finish(self, *, success: bool, error_type: str | None = None, error: str | None = None) -> None:
    self.termination = {
      "at": _now(), "success": success, "error_type": error_type,
      "error": redact(error) if error else None,
    }

  def to_dict(self) -> dict[str, Any]:
    return redact(asdict(self))

  @classmethod
  def from_dict(cls, value: dict[str, Any]) -> "ExecutionTrace":
    known = {key: value[key] for key in cls.__dataclass_fields__ if key in value}
    return cls(**known)
