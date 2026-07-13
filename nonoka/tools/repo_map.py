"""Repo map tools for summarising a codebase symbol tree."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import structlog
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path
from typing import Any

from nonoka.core.context import RunContext
from nonoka.core.tool import tool

logger = structlog.get_logger("nonoka.tools.repo_map")


SKIP_DIRS = frozenset({
  ".git", "node_modules", ".venv", "__pycache__", "dist", "build",
  ".pytest_cache", ".mypy_cache", ".tox", ".eggs", "*.egg-info",
})

SOURCE_EXTS = frozenset({
  ".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java", ".c", ".cpp",
  ".h", ".hpp", ".cs", ".swift", ".kt", ".kts", ".scala", ".rb", ".php",
})

# ---------------------------------------------------------------------------
# Symbol extraction
# ---------------------------------------------------------------------------


def _python_symbols(source: str) -> list[dict[str, Any]]:
  symbols: list[dict[str, Any]] = []
  class_re = re.compile(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:\([^)]*\))?\s*:")
  func_re = re.compile(r"^\s*(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)(?:\s*->[^:]*)?:")
  var_re = re.compile(r"^([A-Z_][A-Z0-9_]*)\s*=")

  for lineno, line in enumerate(source.splitlines(), start=1):
    if m := class_re.match(line):
      symbols.append({
        "kind": "class",
        "name": m.group(1),
        "line": lineno,
        "signature": "",
      })
    elif m := func_re.match(line):
      sig = f"({m.group(2)})"
      symbols.append({
        "kind": "function",
        "name": m.group(1),
        "line": lineno,
        "signature": sig,
      })
    elif m := var_re.match(line):
      symbols.append({
        "kind": "variable",
        "name": m.group(1),
        "line": lineno,
        "signature": "",
      })
  return symbols


def _js_ts_symbols(source: str) -> list[dict[str, Any]]:
  symbols: list[dict[str, Any]] = []
  class_re = re.compile(r"^\s*(?:export\s+(?:default\s+)?)?class\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:extends\s+\S+)?\s*\{")
  func_re = re.compile(r"^\s*(?:export\s+(?:default\s+)?)?(?:async\s+)?function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)")
  arrow_re = re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:async\s+)?(?:\([^)]*\)|[A-Za-z_][A-Za-z0-9_]*)\s*=>")
  method_re = re.compile(r"^\s*(?:async\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)\s*\{")

  for lineno, line in enumerate(source.splitlines(), start=1):
    if m := class_re.match(line):
      symbols.append({"kind": "class", "name": m.group(1), "line": lineno, "signature": ""})
    elif m := func_re.match(line):
      symbols.append({"kind": "function", "name": m.group(1), "line": lineno, "signature": f"({m.group(2)})"})
    elif m := arrow_re.match(line):
      symbols.append({"kind": "function", "name": m.group(1), "line": lineno, "signature": "(...) =>"})
    elif m := method_re.match(line):
      symbols.append({"kind": "method", "name": m.group(1), "line": lineno, "signature": f"({m.group(2)})"})
  return symbols


def _c_family_symbols(source: str) -> list[dict[str, Any]]:
  symbols: list[dict[str, Any]] = []
  func_re = re.compile(r"^\s*(?:[A-Za-z_][A-Za-z0-9_\*\s<>]*\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*\(([^;]*)\)\s*\{?\s*$")
  struct_re = re.compile(r"^\s*(?:typedef\s+)?(?:struct|class|interface|enum)\s+([A-Za-z_][A-Za-z0-9_]*)\b")

  for lineno, line in enumerate(source.splitlines(), start=1):
    if m := struct_re.match(line):
      symbols.append({"kind": "class", "name": m.group(1), "line": lineno, "signature": ""})
    elif m := func_re.match(line):
      name = m.group(1)
      if name in {"if", "for", "while", "switch", "catch"}:
        continue
      symbols.append({"kind": "function", "name": name, "line": lineno, "signature": f"({m.group(2)})"})
  return symbols


def _go_symbols(source: str) -> list[dict[str, Any]]:
  symbols: list[dict[str, Any]] = []
  func_re = re.compile(r"^\s*func\s+(?:\([^)]+\)\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)")
  type_re = re.compile(r"^\s*type\s+([A-Za-z_][A-Za-z0-9_]*)\s+(?:struct|interface)\b")

  for lineno, line in enumerate(source.splitlines(), start=1):
    if m := type_re.match(line):
      symbols.append({"kind": "class", "name": m.group(1), "line": lineno, "signature": ""})
    elif m := func_re.match(line):
      symbols.append({"kind": "function", "name": m.group(1), "line": lineno, "signature": f"({m.group(2)})"})
  return symbols


def _rust_symbols(source: str) -> list[dict[str, Any]]:
  symbols: list[dict[str, Any]] = []
  func_re = re.compile(r"^\s*(?:pub\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)")
  struct_re = re.compile(r"^\s*(?:pub\s+)?(?:struct|trait|enum|impl)\s+(?:<[^>]+>\s+)?([A-Za-z_][A-Za-z0-9_]*)\b")

  for lineno, line in enumerate(source.splitlines(), start=1):
    if m := struct_re.match(line):
      symbols.append({"kind": "class", "name": m.group(1), "line": lineno, "signature": ""})
    elif m := func_re.match(line):
      symbols.append({"kind": "function", "name": m.group(1), "line": lineno, "signature": f"({m.group(2)})"})
  return symbols


def _ruby_symbols(source: str) -> list[dict[str, Any]]:
  symbols: list[dict[str, Any]] = []
  class_re = re.compile(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)")
  func_re = re.compile(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_!?]*)(?:\s*\((.*?)\))?")

  for lineno, line in enumerate(source.splitlines(), start=1):
    if m := class_re.match(line):
      symbols.append({"kind": "class", "name": m.group(1), "line": lineno, "signature": ""})
    elif m := func_re.match(line):
      sig = f"({m.group(2)})" if m.group(2) else ""
      symbols.append({"kind": "function", "name": m.group(1), "line": lineno, "signature": sig})
  return symbols


def _extract_with_regex(path: Path, source: str) -> list[dict[str, Any]]:
  suffix = path.suffix.lower()
  if suffix == ".py":
    return _python_symbols(source)
  if suffix in {".js", ".jsx", ".ts", ".tsx"}:
    return _js_ts_symbols(source)
  if suffix == ".go":
    return _go_symbols(source)
  if suffix == ".rs":
    return _rust_symbols(source)
  if suffix == ".rb":
    return _ruby_symbols(source)
  if suffix in {".java", ".c", ".cpp", ".h", ".hpp", ".cs", ".swift", ".kt", ".kts", ".scala", ".php"}:
    return _c_family_symbols(source)
  return []


def _extract_symbols_tree_sitter(path: Path, source: str) -> list[dict[str, Any]] | None:
  """Extract symbols using tree-sitter, preferring the maintained language pack."""
  suffix = path.suffix.lower()
  lang_map = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".cs": "c_sharp",
    ".rb": "ruby",
    ".php": "php",
  }
  lang = lang_map.get(suffix)
  if lang is None:
    return None

  parser = _get_tree_sitter_parser(lang)
  if parser is None:
    return None

  source_bytes = source.encode("utf-8")
  try:
    tree = parser.parse(source_bytes)
  except Exception:
    return None

  root = tree.root_node
  symbols: list[dict[str, Any]] = []

  kind_map = {
    "class_definition": "class",
    "function_definition": "function",
    "function_declaration": "function",
    "method_definition": "method",
    "method_declaration": "method",
    "class_declaration": "class",
    "struct_item": "class",
    "struct_specifier": "class",
    "interface_declaration": "class",
    "enum_declaration": "class",
  }

  def _node_text(node: Any) -> str:
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

  def signature_for(node: Any) -> str:
    parts: list[str] = []
    for child in node.children:
      if child.type in {"parameters", "formal_parameters", "parameter_list"}:
        parts.append(_node_text(child))
        break
    return " ".join(parts)

  def walk(node: Any) -> None:
    kind = kind_map.get(node.type)
    if kind:
      name = ""
      for child in node.children:
        if child.type in {"identifier", "type_identifier"}:
          name = _node_text(child)
          break
      if name:
        symbols.append({
          "kind": kind,
          "name": name,
          "line": node.start_point[0] + 1,
          "signature": signature_for(node),
        })
    for child in node.children:
      walk(child)

  walk(root)
  return symbols


def _get_tree_sitter_parser(lang: str) -> Any | None:
  """Return a tree-sitter parser for *lang*, trying multiple language packs."""
  # Prefer the maintained tree-sitter-language-pack.
  try:
    from tree_sitter_language_pack import get_parser
    return get_parser(lang)
  except Exception:
    pass

  # Fallback to the older tree-sitter-languages package.
  try:
    from tree_sitter_languages import get_parser  # type: ignore
    return get_parser(lang)
  except Exception:
    pass

  return None


def _extract_symbols_ctags(path: Path) -> list[dict[str, Any]] | None:
  if not shutil.which("ctags"):
    return None
  try:
    import subprocess
    proc = subprocess.run(
      ["ctags", "--fields=+S", "--output-format=json", "--languages=all", str(path)],
      capture_output=True,
      text=True,
      check=False,
    )
    if proc.returncode != 0:
      return None
    stdout = proc.stdout
  except Exception:
    return None

  kind_map = {
    "c": "class",
    "f": "function",
    "m": "method",
    "v": "variable",
    "g": "class",
    "i": "class",
    "t": "class",
  }

  symbols: list[dict[str, Any]] = []
  for line in stdout.decode("utf-8", errors="ignore").splitlines():
    if not line.strip():
      continue
    try:
      entry = json.loads(line)
    except json.JSONDecodeError:
      continue
    symbols.append({
      "kind": kind_map.get(entry.get("kind", ""), "function"),
      "name": entry.get("name", ""),
      "line": entry.get("line", 0),
      "signature": entry.get("signature", ""),
    })
  return symbols


def _extract_symbols(path: Path) -> list[dict[str, Any]]:
  try:
    source = path.read_text(encoding="utf-8", errors="ignore")
  except Exception:
    return []

  symbols = _extract_symbols_tree_sitter(path, source)
  if symbols is None:
    symbols = _extract_symbols_ctags(path)
  if symbols is None:
    symbols = _extract_with_regex(path, source)

  # Tree-sitter surfaces classes/functions/methods but often omits module-level
  # constants (e.g. ``GLOBAL = 42``). Use regex as a supplement to catch those
  # without duplicating symbols already found by tree-sitter/ctags.
  regex_symbols = _extract_with_regex(path, source)
  seen_names = {sym["name"] for sym in symbols}
  for sym in regex_symbols:
    name = sym.get("name")
    if name and name not in seen_names:
      symbols.append(sym)
      seen_names.add(name)

  # Filter out symbols without a valid name.
  return [
    sym for sym in symbols
    if sym.get("name")
  ]


def _lsp_kind_to_symbol_kind(kind: int) -> str:
  """Map an LSP SymbolKind to the repo-map taxonomy."""
  if kind in (5, 11, 23):  # Class, Interface, Struct
    return "class"
  if kind in (6, 9):  # Method, Constructor
    return "method"
  if kind == 12:
    return "function"
  if kind in (13, 8, 7):  # Variable, Field, Property
    return "variable"
  return "function"


def _unified_symbol_to_dict(sym: Any) -> dict[str, Any]:
  """Convert a multilspy UnifiedSymbolInformation to our symbol format."""
  name = sym.get("name")
  if not name:
    return {}
  # Prefer the selection range (the symbol name) over the full declaration range.
  for rng_key in ("selectionRange", "range"):
    rng = sym.get(rng_key, {})
    if isinstance(rng, dict):
      start = rng.get("start", {})
      if isinstance(start, dict) and "line" in start:
        line = start.get("line", 0) + 1
        break
  else:
    line = 0
  return {
    "kind": _lsp_kind_to_symbol_kind(sym.get("kind", 0)),
    "name": name,
    "line": line,
    "signature": sym.get("detail", ""),
  }


def _extract_symbols_lsp_batch(
  files: list[Path],
  lsp_languages: dict[str, str],
  working_dir: Path,
) -> dict[Path, list[dict[str, Any]] | None]:
  """Extract symbols for *files* using language servers.

  Files are grouped by language so that a single language server instance can
  serve all files of that language. Results for files that cannot be handled
  by LSP are left as ``None`` so the caller can fall back to tree-sitter/etc.
  """
  result: dict[Path, list[dict[str, Any]] | None] = {f: None for f in files}
  try:
    from multilspy import SyncLanguageServer
    from multilspy.multilspy_config import MultilspyConfig
    from multilspy.multilspy_logger import MultilspyLogger
  except Exception:
    return result

  groups: dict[str, list[Path]] = {}
  for fpath in files:
    lang = lsp_languages.get(fpath.suffix.lower())
    if lang:
      groups.setdefault(lang, []).append(fpath)

  for lang, group_files in groups.items():
    try:
      config = MultilspyConfig.from_dict({"code_language": lang})
      logger = MultilspyLogger()
      lsp = SyncLanguageServer.create(config, logger, str(working_dir))
      with lsp.start_server():
        for fpath in group_files:
          try:
            rel_path = fpath.relative_to(working_dir).as_posix()
            symbols, _tree = lsp.request_document_symbols(rel_path)
            result[fpath] = [
              s for s in (_unified_symbol_to_dict(s) for s in symbols)
              if s
            ]
          except Exception:
            result[fpath] = None
    except Exception:
      pass

  return result


# ---------------------------------------------------------------------------
# Repo discovery and formatting
# ---------------------------------------------------------------------------


def _is_source_file(path: Path) -> bool:
  return path.suffix.lower() in SOURCE_EXTS and path.is_file()


def _discover_files(root: Path) -> list[Path]:
  files: list[Path] = []
  for dirpath, dirnames, filenames in os.walk(root):
    dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
    for fname in filenames:
      fpath = Path(dirpath) / fname
      if _is_source_file(fpath):
        files.append(fpath)
  return sorted(files)


def _format_repo_map(entries: list[dict[str, Any]], root: Path, max_tokens: int) -> str:
  char_limit = max_tokens * 4
  lines: list[str] = []
  current_dir: str | None = None

  for entry in entries:
    rel = entry["path"]
    parent = str(Path(rel).parent)
    if parent == ".":
      parent = ""
    if parent != current_dir:
      current_dir = parent
      if current_dir:
        lines.append(f"{current_dir}/")
      else:
        lines.append("./")

    fname = Path(rel).name
    lines.append(f"  {fname}:")
    for sym in entry.get("symbols", []):
      name = sym["name"]
      kind = sym.get("kind", "function")
      sig = sym.get("signature", "")
      if kind == "class":
        lines.append(f"    class {name}")
      elif kind == "method":
        lines.append(f"      method {name}{sig}")
      elif kind == "variable":
        lines.append(f"    variable {name}")
      else:
        lines.append(f"    function {name}{sig}")

  text = "\n".join(lines)
  if len(text) > char_limit:
    text = text[:char_limit].rsplit("\n", 1)[0] + "\n... (truncated)"
  return text


def _cache_path(working_dir: Path) -> Path:
  return working_dir / ".nonoka" / "repo_map.jsonl"


def _load_cache(working_dir: Path) -> list[dict[str, Any]] | None:
  cache = _cache_path(working_dir)
  if not cache.exists():
    return None
  entries: list[dict[str, Any]] = []
  try:
    with cache.open("r", encoding="utf-8") as f:
      for line in f:
        line = line.strip()
        if not line:
          continue
        entries.append(json.loads(line))
  except Exception:
    return None
  return entries


def _is_cache_fresh(root: Path, entries: list[dict[str, Any]]) -> bool:
  """Return True if every cached file still exists with the same mtime."""
  for entry in entries:
    path = root / entry.get("path", "")
    if not path.is_file():
      return False
    try:
      if path.stat().st_mtime != entry.get("mtime"):
        return False
    except Exception:
      return False
  return True


def _save_cache(working_dir: Path, entries: list[dict[str, Any]]) -> None:
  cache = _cache_path(working_dir)
  cache.parent.mkdir(parents=True, exist_ok=True)
  with cache.open("w", encoding="utf-8") as f:
    for entry in entries:
      f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _build_repo_map_sync(
  working_dir: Path,
  rel_path: str,
  force_refresh: bool,
  max_tokens: int,
  lsp_languages: dict[str, str] | None = None,
) -> str:
  root = (working_dir / rel_path).resolve()
  if not root.exists():
    return f"Path not found: {rel_path}"
  if not root.is_dir():
    return f"Path is not a directory: {rel_path}"

  if not force_refresh:
    cached = _load_cache(working_dir)
    if cached is not None and _is_cache_fresh(root, cached):
      return _format_repo_map(cached, root, max_tokens)

  files = _discover_files(root)
  lsp_results: dict[Path, list[dict[str, Any]] | None] = {}
  if lsp_languages:
    # LSP indexing is more accurate but can hang on misconfigured servers. Run
    # it with a tight timeout and fall back to tree-sitter/ctags/regex if it
    # does not finish in time.
    try:
      with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
          _extract_symbols_lsp_batch, files, lsp_languages, working_dir
        )
        lsp_results = future.result(timeout=30)
    except TimeoutError:
      logger.warning("lsp_indexing_timeout", languages=list(lsp_languages.keys()))
      lsp_results = {}
    except Exception as exc:
      logger.warning("lsp_indexing_failed", error=str(exc))
      lsp_results = {}

  entries: list[dict[str, Any]] = []
  for fpath in files:
    rel = fpath.relative_to(root).as_posix()
    symbols = lsp_results.get(fpath)
    if symbols is None:
      symbols = _extract_symbols(fpath)
    if symbols:
      entries.append({
        "path": rel,
        "mtime": fpath.stat().st_mtime,
        "symbols": symbols,
      })

  _save_cache(working_dir, entries)
  return _format_repo_map(entries, root, max_tokens)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
async def build_repo_map(
  ctx: RunContext,
  path: str = ".",
  max_tokens: int = 2048,
  force_refresh: bool = False,
  lsp_languages: dict[str, str] | None = None,
) -> str:
  """Build and return a hierarchical repo map for a directory.

  Discovers source files recursively and extracts class/function/method/variable
  symbols. Results are cached in ``.nonoka/repo_map.jsonl`` under the working
  directory for incremental reuse by ``search_repo_map``.

  Args:
    path: Directory to map, resolved relative to the working directory.
    max_tokens: Approximate token budget; output is truncated near this limit.
    force_refresh: If True, rebuild the cache instead of using the cached map.
    lsp_languages: Optional mapping from file extension (e.g. ``.py``) to a
      multilspy ``code_language`` (e.g. ``python``). When provided, the repo
      map uses language servers for those extensions and falls back to
      tree-sitter/ctags/regex for everything else.

  Returns:
    A formatted repo map string in an Aider-style hierarchical layout.
  """
  working_dir = Path(getattr(ctx.deps, "working_dir", "."))
  return await asyncio.to_thread(
    _build_repo_map_sync, working_dir, path, force_refresh, max_tokens, lsp_languages
  )


@tool
async def search_repo_map(
  ctx: RunContext,
  query: str,
  max_results: int = 10,
) -> str:
  """Search the cached repo map for symbols or files matching a query.

  If no cache exists, the map is built on demand from the working directory.

  Args:
    query: Substring to search for in symbol or file names (case-insensitive).
    max_results: Maximum number of matches to return.

  Returns:
    A formatted list of matching file paths and symbols.
  """
  working_dir = Path(getattr(ctx.deps, "working_dir", "."))

  def search() -> str:
    cache = _load_cache(working_dir)
    if cache is None or not _is_cache_fresh(working_dir, cache):
      cache_root = "."
      cache = []
      for fpath in _discover_files(working_dir / cache_root):
        rel = fpath.relative_to(working_dir).as_posix()
        symbols = _extract_symbols(fpath)
        if symbols:
          cache.append({
            "path": rel,
            "mtime": fpath.stat().st_mtime,
            "symbols": symbols,
          })
      _save_cache(working_dir, cache)

    query_lower = query.lower()
    matches: list[str] = []
    for entry in cache:
      rel = entry["path"]
      if query_lower in rel.lower():
        matches.append(f"{rel}")
      for sym in entry.get("symbols", []):
        if query_lower in sym["name"].lower():
          line = sym.get("line", 0)
          matches.append(f"{rel}:{line}  {sym['kind']} {sym['name']}")
      if len(matches) >= max_results:
        break

    if not matches:
      return f"No matches for '{query}'."
    return "\n".join(matches[:max_results])

  return await asyncio.to_thread(search)
