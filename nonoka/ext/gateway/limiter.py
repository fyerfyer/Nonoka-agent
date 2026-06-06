"""
Rate limiter for Gateway layer.

Protects downstream LLM APIs and tool resources from being overwhelmed.
Independent per-user quotas are supported.

Usage::

    limiter = TokenBucketLimiter(default_rate=10, default_burst=20)
    gateway = Gateway(runner=runner, limiter=limiter)
"""

from __future__ import annotations

import asyncio
from typing import Protocol


class Limiter(Protocol):
  """Protocol for Gateway rate limiters."""

  async def acquire(self, key: str) -> bool:
    """Attempt to acquire a slot for *key*.

    Returns ``True`` if the request is allowed, ``False`` if it should
    be rejected (rate limited).
    """
    ...


class TokenBucketLimiter:
  """In-memory token-bucket rate limiter.

  Each *key* gets its own independent bucket.  This implementation is
  intentionally simple — for distributed deployments, replace with a
  Redis-backed limiter.

  Args:
    default_rate: Tokens added per second.
    default_burst: Maximum token bucket size (allows short bursts).
  """

  def __init__(
    self,
    default_rate: float = 10.0,
    default_burst: float = 20.0,
  ):
    self.default_rate = default_rate
    self.default_burst = default_burst
    self._buckets: dict[str, dict[str, float]] = {}
    self._lock = asyncio.Lock()

  async def acquire(self, key: str) -> bool:
    async with self._lock:
      now = asyncio.get_event_loop().time()
      bucket = self._buckets.get(key)

      if bucket is None:
        bucket = {"tokens": self.default_burst, "last_update": now}
        self._buckets[key] = bucket

      # Add tokens based on elapsed time
      elapsed = now - bucket["last_update"]
      bucket["tokens"] = min(
        self.default_burst,
        bucket["tokens"] + elapsed * self.default_rate,
      )
      bucket["last_update"] = now

      if bucket["tokens"] >= 1.0:
        bucket["tokens"] -= 1.0
        return True

      return False
