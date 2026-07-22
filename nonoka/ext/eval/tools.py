"""Workspace-bounded tools used by headless evaluation."""

from __future__ import annotations

import subprocess
import sys
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from nonoka import tool
from nonoka.core.context import RunContext
from nonoka.core.execution import ToolExecution
from nonoka.ext.coding import WorkspaceAuditor, WorkspaceMutation


@dataclass
class EvalDeps:
  root: Path
  tool_trace: list[str] = field(default_factory=list)
  mutations: list[WorkspaceMutation] = field(default_factory=list)

  def resolve(self, relative_path: str) -> Path:
    candidate = (self.root / relative_path).resolve()
    if candidate != self.root.resolve() and self.root.resolve() not in candidate.parents:
      raise ValueError("Path escapes the evaluation workspace")
    return candidate

  def audit_mutation(self, tool_name: str, before: dict[str, str]) -> None:
    auditor = WorkspaceAuditor(self.root)
    self.mutations.append(WorkspaceMutation(tool_name, auditor.diff(before)))


def get_eval_tools() -> list[object]:
  @tool(execution=ToolExecution(read_only=True))
  def read_file(ctx: RunContext[EvalDeps], path: str) -> str:
    """Read a UTF-8 file relative to the evaluation workspace."""
    ctx.deps.tool_trace.append(f"read_file:{path}")
    return ctx.deps.resolve(path).read_text(encoding="utf-8")

  @tool(execution=ToolExecution(mutates_workspace=True, stateful_action=True))
  def write_file(ctx: RunContext[EvalDeps], path: str, content: str) -> str:
    """Write a UTF-8 file relative to the evaluation workspace."""
    ctx.deps.tool_trace.append(f"write_file:{path}")
    before = WorkspaceAuditor(ctx.deps.root).snapshot()
    target = ctx.deps.resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    ctx.deps.audit_mutation("write_file", before)
    return f"wrote {path}"

  @tool(execution=ToolExecution(mutates_workspace=True, stateful_action=True))
  def delete_file(ctx: RunContext[EvalDeps], path: str) -> str:
    """Delete a file relative to the evaluation workspace."""
    ctx.deps.tool_trace.append(f"delete_file:{path}")
    before = WorkspaceAuditor(ctx.deps.root).snapshot()
    ctx.deps.resolve(path).unlink()
    ctx.deps.audit_mutation("delete_file", before)
    return f"deleted {path}"

  @tool(execution=ToolExecution(read_only=True))
  def list_dir(ctx: RunContext[EvalDeps], path: str = ".") -> str:
    """List files below a workspace-relative directory."""
    ctx.deps.tool_trace.append(f"list_dir:{path}")
    target = ctx.deps.resolve(path)
    return "\n".join(sorted(str(item.relative_to(ctx.deps.root)) for item in target.rglob("*") if item.is_file()))

  @tool(execution=ToolExecution(read_only=True, stateful_action=True))
  def execute_python(ctx: RunContext[EvalDeps], code: str) -> str:
    """Run a short Python command in the evaluation workspace."""
    ctx.deps.tool_trace.append("execute_python")
    # Validation runs against an ephemeral copy.  A model cannot bypass
    # write_file by using Python to alter the actual evaluation workspace.
    # Python 3.14 can leave asyncio's subprocess pipe waiter unresolved even
    # after the child exits, and a thread-pool workaround can hang while
    # asyncio.run() shuts down its executor. This bounded, short subprocess
    # is therefore deliberately synchronous.
    try:
      with tempfile.TemporaryDirectory(prefix="nonoka-eval-verify-") as temp_dir:
        copied_root = Path(temp_dir) / "workspace"
        shutil.copytree(ctx.deps.root, copied_root)
        copied_script = (
          "import sys; "
          f"sys.path.insert(0, {str(copied_root)!r}); "
          f"exec(compile({code!r}, '<eval-tool>', 'exec'))"
        )
        result = subprocess.run(
          [sys.executable, "-I", "-c", copied_script],
          cwd=copied_root,
          stdout=subprocess.PIPE,
          stderr=subprocess.PIPE,
          timeout=10,
          check=False,
        )
    except subprocess.TimeoutExpired as exc:
      stdout = exc.stdout or b""
      stderr = exc.stderr or b""
      return (stdout + stderr + b"\nTimed out after 10 seconds.").decode("utf-8", errors="replace")
    return (result.stdout + result.stderr).decode("utf-8", errors="replace")

  return [read_file, write_file, delete_file, list_dir, execute_python]
