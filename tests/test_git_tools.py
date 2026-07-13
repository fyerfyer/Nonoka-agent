"""Tests for the git checkpoint tools."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from nonoka.core.agent import Agent
from nonoka.core.context import RunContext
from nonoka.core.session import Session
from nonoka.tools.git import git_checkpoint, git_rollback, git_status


async def _sh(cmd: str, cwd: Path) -> tuple[int, str, str]:
  proc = await asyncio.create_subprocess_shell(
    cmd,
    cwd=cwd,
    stdout=asyncio.subprocess.PIPE,
    stderr=asyncio.subprocess.PIPE,
  )
  stdout, stderr = await proc.communicate()
  return (
    proc.returncode,
    stdout.decode("utf-8", errors="replace").strip(),
    stderr.decode("utf-8", errors="replace").strip(),
  )


def _ctx(repo: Path) -> RunContext:
  agent = Agent(model="test")
  session = Session(
    session_id="test-git",
    agent=agent,
    deps=SimpleNamespace(working_dir=str(repo)),
  )
  return RunContext(session)


@pytest.mark.asyncio
async def test_git_checkpoint_and_rollback(tmp_path: Path) -> None:
  repo = tmp_path / "repo"
  repo.mkdir()

  await _sh("git init", repo)
  await _sh("git config user.email 'test@example.com'", repo)
  await _sh("git config user.name 'Test User'", repo)

  # Base commit so HEAD is not empty
  (repo / "base.txt").write_text("base")
  await _sh("git add base.txt && git commit -m 'initial commit'", repo)

  ctx = _ctx(repo)

  # Pre-existing change should be preserved before the checkpoint
  (repo / "pre-existing.txt").write_text("pre-existing")
  result = await git_checkpoint(ctx)
  assert "nonoka checkpoint:" in result

  rc, log, _ = await _sh("git log --author='(nonoka)' --oneline", repo)
  assert rc == 0
  assert len(log.splitlines()) == 2  # preserve + checkpoint

  # Checkpoint with a custom message
  (repo / "checkpoint-a.txt").write_text("a")
  result = await git_checkpoint(ctx, message="my checkpoint a")
  assert "my checkpoint a" in result

  rc, author, _ = await _sh("git log -1 --format='%an %ae'", repo)
  assert rc == 0
  assert "Test User (nonoka)" in author

  # Rollback multiple checkpoint commits
  (repo / "checkpoint-b.txt").write_text("b")
  await git_checkpoint(ctx, message="my checkpoint b")

  rollback = await git_rollback(ctx, steps=2)
  assert "Rolled back to" in rollback
  assert not (repo / "checkpoint-b.txt").exists()
  assert (repo / "checkpoint-a.txt").exists()

  # Status should report cleanly
  status = await git_status(ctx)
  assert "Status:" in status
  assert "HEAD:" in status
  assert "my checkpoint a" in status


@pytest.mark.asyncio
async def test_git_status_not_a_repo(tmp_path: Path) -> None:
  ctx = _ctx(tmp_path)
  result = await git_status(ctx)
  assert "Not a git repository" in result


@pytest.mark.asyncio
async def test_git_rollback_by_hash(tmp_path: Path) -> None:
  repo = tmp_path / "repo"
  repo.mkdir()

  await _sh("git init", repo)
  await _sh("git config user.email 'test@example.com'", repo)
  await _sh("git config user.name 'Test User'", repo)

  (repo / "base.txt").write_text("base")
  await _sh("git add base.txt && git commit -m 'initial commit'", repo)

  ctx = _ctx(repo)
  target = await git_checkpoint(ctx, message="target")
  target_hash = target.split()[0]

  (repo / "extra.txt").write_text("extra")
  await git_checkpoint(ctx, message="extra")

  rollback = await git_rollback(ctx, commit_hash=target_hash)
  assert target_hash in rollback
  assert not (repo / "extra.txt").exists()
