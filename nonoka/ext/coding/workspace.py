"""Filesystem diffing used by coding tools and evaluation extensions."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


def _digest(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class WorkspaceDiff:
  created: tuple[str, ...] = ()
  modified: tuple[str, ...] = ()
  deleted: tuple[str, ...] = ()

  @property
  def changed(self) -> tuple[str, ...]:
    return self.created + self.modified + self.deleted


@dataclass(frozen=True)
class WorkspaceMutation:
  tool: str
  diff: WorkspaceDiff


class WorkspaceAuditor:
  """Take content-hash snapshots of a bounded workspace.

  It is an audit mechanism rather than an OS security sandbox.  Callers that
  execute untrusted code should use a copy or a Docker/VM boundary as well.
  """

  def __init__(self, root: Path) -> None:
    self.root = root.resolve()

  def snapshot(self) -> dict[str, str]:
    state: dict[str, str] = {}
    for item in self.root.rglob("*"):
      if item.is_file() and not item.is_symlink():
        state[str(item.relative_to(self.root))] = _digest(item)
    return state

  def diff(self, before: dict[str, str], after: dict[str, str] | None = None) -> WorkspaceDiff:
    after = self.snapshot() if after is None else after
    return WorkspaceDiff(
      created=tuple(sorted(set(after) - set(before))),
      deleted=tuple(sorted(set(before) - set(after))),
      modified=tuple(sorted(path for path in set(before) & set(after) if before[path] != after[path])),
    )
