"""Language Server Protocol (LSP) tools for precise code intelligence.

These tools are optional. They require the ``multilspy`` package and a
suitable language server binary to be installed for the target language
(e.g. ``jedi-language-server`` for Python).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nonoka.core.context import RunContext
from nonoka.core.tool import tool


_LSP_LANGUAGE_MAP = {
  ".py": "python",
  ".js": "javascript",
  ".jsx": "javascript",
  ".ts": "typescript",
  ".tsx": "typescript",
  ".rs": "rust",
  ".go": "go",
  ".java": "java",
  ".c": "c",
  ".cpp": "cpp",
  ".h": "c",
  ".hpp": "cpp",
  ".cs": "csharp",
  ".rb": "ruby",
  ".php": "php",
}


def _lsp_kind_to_kind(kind: int) -> str:
  """Map LSP SymbolKind values to our simplified symbol kinds."""
  # SymbolKind values are 1-based as defined by LSP.
  if kind in (5, 11, 23):  # Class, Interface, Struct
    return "class"
  if kind in (6, 9):  # Method, Constructor
    return "method"
  if kind == 12:
    return "function"
  if kind in (13, 8, 7):  # Variable, Field, Property
    return "variable"
  return "symbol"


def _format_symbols(symbols: list[Any]) -> str:
  """Render a list of LSP symbols as human-readable text."""
  if not symbols:
    return "No symbols found."
  lines: list[str] = []
  for sym in symbols:
    name = sym.get("name", "")
    if not name:
      continue
    kind = _lsp_kind_to_kind(sym.get("kind", 0))
    detail = sym.get("detail") or ""
    line = 0
    for rng_key in ("selectionRange", "range"):
      rng = sym.get(rng_key, {})
      if isinstance(rng, dict):
        start = rng.get("start", {})
        if isinstance(start, dict) and "line" in start:
          line = start.get("line", 0) + 1
          break
    sig = f" {detail}" if detail else ""
    lines.append(f"{kind} {name}{sig} (line {line})")
  return "\n".join(lines)


@tool
async def lsp_document_symbols(ctx: RunContext, file_path: str) -> str:
  """Return the LSP document symbols for *file_path*.

  Uses a language server appropriate for the file extension. This is useful
  when the regex/tree-sitter repo map is insufficient or when you need exact
  symbol locations for a specific file.

  Args:
    file_path: Path to the source file, relative to the working directory.

  Returns:
    A formatted list of symbols, or an error message.
  """
  try:
    from multilspy import LanguageServer
    from multilspy.multilspy_config import MultilspyConfig
    from multilspy.multilspy_logger import MultilspyLogger
  except Exception as exc:
    return (
      f"LSP support is not available: {exc}. "
      "Install the repo-map extras (e.g. nonoka[repo-map]) to enable it."
    )

  working_dir = Path(getattr(ctx.deps, "working_dir", "."))
  path = working_dir / file_path
  try:
    path = path.resolve().relative_to(working_dir.resolve())
    rel_path = path.as_posix()
  except Exception:
    rel_path = file_path

  suffix = path.suffix.lower() if path.suffix else Path(file_path).suffix.lower()
  lang = _LSP_LANGUAGE_MAP.get(suffix)
  if lang is None:
    return f"No LSP language mapping for extension '{suffix}'."

  try:
    config = MultilspyConfig.from_dict({"code_language": lang})
    logger = MultilspyLogger()
    lsp = LanguageServer.create(config, logger, str(working_dir))
  except Exception as exc:
    return f"Failed to create LSP client for {lang}: {exc}"

  try:
    async with lsp.start_server():
      symbols, _tree = await lsp.request_document_symbols(rel_path)
      return _format_symbols(symbols)
  except Exception as exc:
    return f"LSP document symbol request failed for {file_path}: {exc}"
