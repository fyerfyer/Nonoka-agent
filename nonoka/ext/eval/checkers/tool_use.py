from __future__ import annotations

from pathlib import Path

from nonoka.ext.eval.models import EvalSample
from nonoka.ext.eval.tools import EvalDeps


class ToolUseChecker:
  def check(self, sample: EvalSample, root: Path, deps: EvalDeps) -> tuple[bool, str]:
    if not deps.tool_trace:
      return False, "agent did not call an evaluation tool"
    seen_tools = {event.split(":", 1)[0] for event in deps.tool_trace}
    required_tools = set(sample.metadata.get("required_tools", []))
    missing_tools = sorted(required_tools - seen_tools)
    if missing_tools:
      return False, f"missing required tool use: {', '.join(missing_tools)}"
    mutations_by_tool: dict[str, set[str]] = {}
    for mutation in deps.mutations:
      mutations_by_tool.setdefault(mutation.tool, set()).update(mutation.diff.changed)
    for name, expected in sample.metadata.get("expected", {}).items():
      path = root / name
      if not path.exists():
        return False, f"missing expected file: {name}"
      if path.read_text(encoding="utf-8") != expected:
        return False, f"unexpected content in {name}"
      if name not in mutations_by_tool.get("write_file", set()):
        return False, f"expected file was not created through write_file: {name}"
    for name in sample.metadata.get("absent", []):
      if (root / name).exists():
        return False, f"file should be absent: {name}"
      if name not in mutations_by_tool.get("delete_file", set()):
        return False, f"expected deletion was not created through delete_file: {name}"
    return True, f"workspace verified after {len(deps.tool_trace)} tool calls"
