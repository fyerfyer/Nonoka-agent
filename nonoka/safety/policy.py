"""Static policy checks which fail closed before a tool executes."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from nonoka.core.errors import SafetyError

_DENY = (re.compile(r"\brm\s+-[A-Za-z]*r[A-Za-z]*f\s+/(?:\s|$)", re.I), re.compile(r"\bmkfs\b|\bdd\s+if=.*\bof=/dev/", re.I))
_APPROVE = (re.compile(r"\bsudo\b|\brm\b|\bchmod\b|\bchown\b|\bcurl\b.*\|\s*(?:ba)?sh", re.I),)


@dataclass
class SafetyPolicy:
  allowed_roots: list[Path] = field(default_factory=list)
  denied_names: set[str] = field(default_factory=lambda: {".git", ".env", ".ssh"})

  def check_command(self, command: str) -> str:
    if any(pattern.search(command) for pattern in _DENY):
      raise SafetyError("Command denied by safety policy")
    return "approval" if any(pattern.search(command) for pattern in _APPROVE) else "allow"

  def check_path(self, path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve(strict=False)
    if any(part in self.denied_names for part in resolved.parts):
      raise SafetyError(f"Path denied by safety policy: {resolved}")
    if self.allowed_roots and not any(resolved.is_relative_to(root.resolve()) for root in self.allowed_roots):
      raise SafetyError(f"Path outside allowed roots: {resolved}")
    return resolved
