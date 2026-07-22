"""Deterministic final-response grounding for the τ³ adapter."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


_NUMBER = re.compile(r"(?<![\w.-])-?\d+(?:\.\d+)?(?![\w.-])")
_STATE_WORDS = {"available", "unavailable", "in stock", "out of stock", "cancelled", "canceled", "refunded", "shipped", "pending"}
_QUANTITY_CUES = {"available", "availability", "stock", "sku", "item", "items", "option", "options", "quantity", "remaining", "库存", "可用"}


@dataclass(frozen=True)
class EvidenceFact:
  path: str
  value: str


@dataclass(frozen=True)
class GroundingFinding:
  passed: bool
  feedback: str = ""
  fact_paths: tuple[str, ...] = ()

  def to_dict(self) -> dict[str, Any]:
    return {"passed": self.passed, "feedback": self.feedback, "fact_paths": list(self.fact_paths)}


def validate_final_response(messages: list[Any], content: str, policy: str = "") -> GroundingFinding:
  """Reject factual quantity/state claims contradicted by tool observations.

  The check intentionally validates only facts with a recognisable domain cue.
  It does not try to judge general prose, and policy-provided numerical rules
  are accepted without a tool lookup.
  """
  facts = _facts_from_messages(messages)
  policy_numbers = set(_NUMBER.findall(policy))
  issues: list[str] = []
  paths: list[str] = []
  lowered = content.lower()
  for match in _NUMBER.finditer(content):
    window = lowered[max(0, match.start() - 36):match.end() + 36]
    cues = {cue for cue in _QUANTITY_CUES if cue in window}
    if not cues or match.group(0) in policy_numbers:
      continue
    candidates = [fact for fact in facts if any(cue in fact.path.lower() for cue in cues)]
    if candidates and match.group(0) not in {fact.value for fact in candidates}:
      issues.append(
        f"The claimed value {match.group(0)!r} conflicts with tool evidence for {', '.join(sorted(cues))}."
      )
      paths.extend(fact.path for fact in candidates)
  for state in _STATE_WORDS:
    if state not in lowered:
      continue
    state_facts = [fact for fact in facts if fact.value.lower() in _STATE_WORDS]
    if state_facts and state not in {fact.value.lower() for fact in state_facts}:
      issues.append(f"The claimed state {state!r} conflicts with the latest tool state.")
      paths.extend(fact.path for fact in state_facts)
  if not issues:
    return GroundingFinding(True)
  return GroundingFinding(
    False,
    " ".join(issues) + " Rewrite the response using only the verified tool values; do not call tools in this revision.",
    tuple(sorted(set(paths))),
  )


def _facts_from_messages(messages: list[Any]) -> list[EvidenceFact]:
  facts: list[EvidenceFact] = []
  for message in messages:
    role = str(_field(message, "role") or "")
    if role != "tool":
      continue
    content = _field(message, "content")
    value = _decode_content(content)
    _walk_facts(value, "tool", facts)
  return facts


def _field(value: Any, name: str) -> Any:
  if isinstance(value, dict):
    return value.get(name)
  return getattr(value, name, None)


def _decode_content(value: Any) -> Any:
  if isinstance(value, str):
    try:
      return json.loads(value)
    except json.JSONDecodeError:
      return value
  return value


def _walk_facts(value: Any, path: str, facts: list[EvidenceFact]) -> None:
  if isinstance(value, dict):
    for key, child in value.items():
      _walk_facts(child, f"{path}.{key}", facts)
  elif isinstance(value, list):
    for index, child in enumerate(value):
      _walk_facts(child, f"{path}[{index}]", facts)
  elif isinstance(value, (str, int, float, bool)):
    facts.append(EvidenceFact(path, str(value)))
