"""Small, local response cache with safe exact-match semantics."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Protocol

from nonoka.core.llm import LLMResponse


def canonical_response_key(
  *, model: str, messages: list[Any], tools: list[dict[str, Any]] | None,
  temperature: float | None, max_tokens: int | None, namespace: str = "default",
) -> str:
  payload = {
    "v": 1, "namespace": namespace, "model": model,
    "messages": [item.model_dump(exclude_none=True) if hasattr(item, "model_dump") else item for item in messages],
    "tools": tools or [], "temperature": temperature, "max_tokens": max_tokens,
  }
  raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
  return hashlib.sha256(raw.encode()).hexdigest()


def _message_field(message: Any, field: str, default: Any = None) -> Any:
  if hasattr(message, field):
    return getattr(message, field)
  if isinstance(message, dict):
    return message.get(field, default)
  return default


def semantic_cache_query(messages: list[Any]) -> str | None:
  """Return a safe single-turn query, or ``None`` for contextual exchanges.

  Semantic response reuse cannot safely infer whether an answer remains valid
  after previous assistant/tool turns.  It is therefore deliberately limited
  to a stateless request: one user message plus optional system instructions.
  System instructions are represented by the cache scope rather than embedded
  again, avoiding volatile prompt metadata and large duplicate vectors.
  """
  user_content: str | None = None
  for message in messages:
    role = str(_message_field(message, "role", "")).lower()
    if role.startswith("llmmessagerole."):
      role = role.removeprefix("llmmessagerole.")
    if role == "system":
      continue
    if role != "user" or user_content is not None:
      return None
    if _message_field(message, "name") or _message_field(message, "tool_call_id") or _message_field(message, "tool_calls"):
      return None
    content = _message_field(message, "content")
    if not isinstance(content, str) or not content.strip():
      return None
    user_content = content.strip()
  return user_content


def semantic_cache_variant(*, model: str, temperature: float | None, max_tokens: int | None) -> str:
  """Fingerprint response-shaping settings which embedding similarity cannot encode."""
  raw = json.dumps({"v": 1, "model": model, "temperature": temperature, "max_tokens": max_tokens}, sort_keys=True, separators=(",", ":"))
  return hashlib.sha256(raw.encode()).hexdigest()


class ResponseCache(Protocol):
  async def get(self, key: str) -> LLMResponse | None: ...
  async def put(self, key: str, response: LLMResponse, ttl_seconds: int) -> None: ...


class SQLiteResponseCache:
  """SQLite cache deliberately limited to complete, tool-free responses."""

  def __init__(self, path: str | Path) -> None:
    self.path = Path(path).expanduser()
    self._init_lock = asyncio.Lock()
    self._initialized = False

  async def _init(self) -> None:
    if self._initialized:
      return
    async with self._init_lock:
      if self._initialized:
        return
      self.path.parent.mkdir(parents=True, exist_ok=True)
      with sqlite3.connect(self.path) as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("CREATE TABLE IF NOT EXISTS llm_response_cache (key TEXT PRIMARY KEY, response TEXT NOT NULL, expires_at REAL NOT NULL)")
        db.execute("CREATE INDEX IF NOT EXISTS llm_response_cache_expires ON llm_response_cache(expires_at)")
      self._initialized = True

  async def get(self, key: str) -> LLMResponse | None:
    await self._init()
    def read() -> LLMResponse | None:
      with sqlite3.connect(self.path) as db:
        row = db.execute("SELECT response, expires_at FROM llm_response_cache WHERE key = ?", (key,)).fetchone()
        if row is None:
          return None
        if row[1] <= time.time():
          db.execute("DELETE FROM llm_response_cache WHERE key = ?", (key,))
          return None
        return LLMResponse.model_validate_json(row[0])
    return await asyncio.to_thread(read)

  async def put(self, key: str, response: LLMResponse, ttl_seconds: int) -> None:
    await self._init()
    payload = response.model_dump_json()
    expires_at = time.time() + ttl_seconds
    def write() -> None:
      with sqlite3.connect(self.path) as db:
        db.execute("INSERT OR REPLACE INTO llm_response_cache(key, response, expires_at) VALUES (?, ?, ?)", (key, payload, expires_at))
    await asyncio.to_thread(write)
