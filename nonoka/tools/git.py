"""Built-in tools for git checkpoint / rollback."""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone

from nonoka.core.context import RunContext
from nonoka.core.tool import tool


async def _run_git(
  cmd: list[str],
  cwd: str,
  env: dict[str, str] | None = None,
) -> tuple[int, str, str]:
  """Run a git command and return (returncode, stdout, stderr)."""
  proc = await asyncio.create_subprocess_exec(
    *cmd,
    cwd=cwd,
    env=env,
    stdout=asyncio.subprocess.PIPE,
    stderr=asyncio.subprocess.PIPE,
  )
  stdout, stderr = await proc.communicate()
  return (
    proc.returncode,
    stdout.decode("utf-8", errors="replace").strip(),
    stderr.decode("utf-8", errors="replace").strip(),
  )


async def _repo_root(cwd: str) -> tuple[bool, str]:
  """Verify *cwd* is inside a git repository."""
  rc, _, err = await _run_git(["git", "rev-parse", "--git-dir"], cwd)
  if rc != 0:
    return False, f"Not a git repository: {err or 'unknown error'}"
  return True, ""


async def _nonoka_author(cwd: str) -> tuple[tuple[str, str] | None, str]:
  """Return (author_name, author_email) with ``(nonoka)`` appended to name."""
  rc, name, _ = await _run_git(["git", "config", "user.name"], cwd)
  if rc != 0 or not name:
    return None, "git user.name is not configured"
  rc_email, email, _ = await _run_git(["git", "config", "user.email"], cwd)
  if rc_email != 0 or not email:
    # Git requires an email to commit; synthesize a stable fallback so the
    # checkpoint does not fail in environments where user.email is unset.
    email = f"{name.replace(' ', '.').lower()}@nonoka.local"
  marked_name = f"{name} (nonoka)"
  return (marked_name, email), ""


async def _git_env(cwd: str) -> tuple[dict[str, str] | None, str]:
  """Build an environment dict that tags commits as authored by nonoka."""
  author, err = await _nonoka_author(cwd)
  if author is None:
    return None, err
  author_name, author_email = author
  env = os.environ.copy()
  env["GIT_AUTHOR_NAME"] = author_name
  env["GIT_COMMITTER_NAME"] = author_name
  env["GIT_AUTHOR_EMAIL"] = author_email
  env["GIT_COMMITTER_EMAIL"] = author_email
  return env, ""


async def _is_dirty(cwd: str) -> bool:
  rc, out, _ = await _run_git(["git", "status", "--porcelain"], cwd)
  return rc == 0 and bool(out)


async def _commit_all(
  cwd: str,
  message: str,
  env: dict[str, str],
  allow_empty: bool = False,
) -> tuple[int, str, str]:
  await _run_git(["git", "add", "-A"], cwd, env)
  cmd = ["git", "commit", "-m", message]
  if allow_empty:
    cmd.append("--allow-empty")
  return await _run_git(cmd, cwd, env)


async def _head(cwd: str) -> str:
  rc, out, _ = await _run_git(["git", "rev-parse", "HEAD"], cwd)
  return out if rc == 0 else "unknown"


@tool
async def git_checkpoint(ctx: RunContext, message: str | None = None) -> str:
  """Create a checkpoint commit in the current git repository.

  If the working tree already contains uncommitted changes, those changes are
  committed first with a preservation message. Then all current changes are
  staged and committed as a checkpoint authored by ``<user> (nonoka)``.

  Args:
    message: Optional commit message. Defaults to ``nonoka checkpoint: <iso>``.

  Returns:
    The checkpoint commit hash and message, or an error string.
  """
  cwd = ctx.deps.working_dir

  is_repo, err = await _repo_root(cwd)
  if not is_repo:
    return f"Error: {err}"

  env, err = await _git_env(cwd)
  if env is None:
    return f"Error: {err}"

  if await _is_dirty(cwd):
    rc, _, err = await _commit_all(
      cwd,
      "nonoka: preserve pre-existing changes before checkpoint",
      env,
    )
    if rc != 0:
      return f"Error preserving pre-existing changes: {err}"

  checkpoint_message = message or (
    f"nonoka checkpoint: {datetime.now(timezone.utc).isoformat()}"
  )

  rc, _, err = await _commit_all(
    cwd,
    checkpoint_message,
    env,
    allow_empty=True,
  )
  if rc != 0:
    return f"Error creating checkpoint: {err}"

  head = await _head(cwd)
  return f"{head} {checkpoint_message}"


@tool
async def git_rollback(
  ctx: RunContext,
  commit_hash: str | None = None,
  steps: int = 1,
) -> str:
  """Roll back to a previous checkpoint commit.

  Args:
    commit_hash: Optional hash to reset to directly.
    steps: Number of checkpoint commits to roll back when *commit_hash* is not
      provided.

  Returns:
    The new HEAD and a summary, or an error string.
  """
  cwd = ctx.deps.working_dir

  is_repo, err = await _repo_root(cwd)
  if not is_repo:
    return f"Error: {err}"

  if steps < 1:
    return "Error: steps must be >= 1"

  if commit_hash is None:
    # To roll back N steps we need the (N+1)-th most recent checkpoint
    # (the commit N steps behind HEAD).
    rc, out, err = await _run_git(
      ["git", "log", "--author=(nonoka)", "--format=%H", f"-n{steps + 1}"],
      cwd,
    )
    if rc != 0:
      return f"Error reading checkpoint history: {err}"

    hashes = [h for h in out.splitlines() if h]
    if len(hashes) < steps + 1:
      return f"Error: not enough checkpoint commits to roll back {steps} step(s)"

    target = hashes[-1]
  else:
    target = commit_hash

  rc, _, err = await _run_git(["git", "reset", "--hard", target], cwd)
  if rc != 0:
    return f"Error rolling back to {target}: {err}"

  head = await _head(cwd)
  return f"Rolled back to {target}. New HEAD: {head}"


@tool
async def git_status(ctx: RunContext) -> str:
  """Return the git status, recent checkpoint commits, and current HEAD."""
  cwd = ctx.deps.working_dir

  is_repo, err = await _repo_root(cwd)
  if not is_repo:
    return f"Error: {err}"

  rc, status, _ = await _run_git(["git", "status", "--short"], cwd)
  if rc != 0:
    status = f"Error reading status: {status}"

  rc, log, _ = await _run_git(
    ["git", "log", "--author=(nonoka)", "--oneline", "-n10"],
    cwd,
  )
  if rc != 0:
    log = f"Error reading checkpoint log: {log}"

  head = await _head(cwd)

  return (
    f"Status:\n{status or '(clean)'}\n\n"
    f"Recent checkpoint commits:\n{log or '(none)'}\n\n"
    f"HEAD: {head}"
  )
