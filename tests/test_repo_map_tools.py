"""Tests for the repo map tools."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from nonoka.core.agent import Agent
from nonoka.core.context import RunContext
from nonoka.core.session import Session
from nonoka.tools.repo_map import build_repo_map, search_repo_map


def _ctx(working_dir: Path) -> RunContext:
  agent = Agent(model="test")
  session = Session(
    session_id="test-repo-map",
    agent=agent,
    deps=SimpleNamespace(working_dir=str(working_dir)),
  )
  return RunContext(session)


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
  repo = tmp_path / "repo"
  repo.mkdir()

  src = repo / "src"
  src.mkdir()

  (src / "foo.py").write_text(
    "class Bar:\n"
    "  def baz(self, x: int) -> str:\n"
    "    return str(x)\n"
    "\n"
    "GLOBAL = 42\n"
  )

  (src / "helpers.js").write_text(
    "function qux(a, b) {\n"
    "  return a + b;\n"
    "}\n"
    "\n"
    "class Widget {\n"
    "  render() {\n"
    "    return null;\n"
    "  }\n"
    "}\n"
  )

  # Should be skipped
  (src / "__pycache__").mkdir()
  (src / "__pycache__" / "cache.pyc").write_text("ignored")

  return repo


@pytest.mark.asyncio
async def test_build_repo_map_extracts_python_and_js_symbols(sample_repo: Path) -> None:
  ctx = _ctx(sample_repo)
  result = await build_repo_map(ctx, path=".")

  assert isinstance(result, str)
  assert "class Bar" in result
  assert "method baz" in result or "function baz" in result
  assert "variable GLOBAL" in result
  assert "function qux" in result
  assert "class Widget" in result
  assert "render" in result

  # Cache file created
  cache = sample_repo / ".nonoka" / "repo_map.jsonl"
  assert cache.exists()


@pytest.mark.asyncio
async def test_build_repo_map_uses_cache_on_second_call(sample_repo: Path) -> None:
  ctx = _ctx(sample_repo)
  first = await build_repo_map(ctx, path=".")
  second = await build_repo_map(ctx, path=".")
  assert first == second


@pytest.mark.asyncio
async def test_search_repo_map_finds_symbols_and_files(sample_repo: Path) -> None:
  ctx = _ctx(sample_repo)
  result = await search_repo_map(ctx, query="qux")
  assert isinstance(result, str)
  assert "qux" in result

  result = await search_repo_map(ctx, query="Bar")
  assert "Bar" in result

  result = await search_repo_map(ctx, query="helpers.js")
  assert "helpers.js" in result


@pytest.mark.asyncio
async def test_search_repo_map_no_matches(sample_repo: Path) -> None:
  ctx = _ctx(sample_repo)
  result = await search_repo_map(ctx, query="definitely_missing_symbol")
  assert "No matches" in result
