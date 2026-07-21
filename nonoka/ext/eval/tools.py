"""Workspace-bounded tools used by headless evaluation."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from nonoka import tool
from nonoka.core.context import RunContext


@dataclass
class EvalDeps:
  root: Path
  tool_trace: list[str] = field(default_factory=list)

  def resolve(self, relative_path: str) -> Path:
    candidate = (self.root / relative_path).resolve()
    if candidate != self.root.resolve() and self.root.resolve() not in candidate.parents:
      raise ValueError("Path escapes the evaluation workspace")
    return candidate


def get_eval_tools() -> list[object]:
  @tool
  def read_file(ctx: RunContext[EvalDeps], path: str) -> str:
    """Read a UTF-8 file relative to the evaluation workspace."""
    ctx.deps.tool_trace.append(f"read_file:{path}")
    return ctx.deps.resolve(path).read_text(encoding="utf-8")

  @tool
  def write_file(ctx: RunContext[EvalDeps], path: str, content: str) -> str:
    """Write a UTF-8 file relative to the evaluation workspace."""
    ctx.deps.tool_trace.append(f"write_file:{path}")
    target = ctx.deps.resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"wrote {path}"

  @tool
  def delete_file(ctx: RunContext[EvalDeps], path: str) -> str:
    """Delete a file relative to the evaluation workspace."""
    ctx.deps.tool_trace.append(f"delete_file:{path}")
    ctx.deps.resolve(path).unlink()
    return f"deleted {path}"

  @tool
  def list_dir(ctx: RunContext[EvalDeps], path: str = ".") -> str:
    """List files below a workspace-relative directory."""
    ctx.deps.tool_trace.append(f"list_dir:{path}")
    target = ctx.deps.resolve(path)
    return "\n".join(sorted(str(item.relative_to(ctx.deps.root)) for item in target.rglob("*") if item.is_file()))

  @tool
  def execute_python(ctx: RunContext[EvalDeps], code: str) -> str:
    """Run a short Python command in the evaluation workspace."""
    ctx.deps.tool_trace.append("execute_python")
    # Keep isolated mode, but expose only this temporary workspace so agents
    # can validate the solution.py they just wrote.
    script = (
      "import sys; "
      f"sys.path.insert(0, {str(ctx.deps.root)!r}); "
      f"exec(compile({code!r}, '<eval-tool>', 'exec'))"
    )
    # Python 3.14 can leave asyncio's subprocess pipe waiter unresolved even
    # after the child exits, and a thread-pool workaround can hang while
    # asyncio.run() shuts down its executor. This bounded, short subprocess
    # is therefore deliberately synchronous.
    try:
      result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=ctx.deps.root,
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
