import json
import asyncio
import traceback as _traceback
from typing import Any
from nonoka.core.checkpoint import CheckpointStore
from nonoka.core.session import SessionState, StepStatus, StepResult, StepFailure

# You might need to install redis package for this to work
# pip install redis
try:
  import redis.asyncio as redis
except ImportError:
  redis = None


class RedisCheckpointStore(CheckpointStore):
  """
  Redis-backed checkpoint store for production use.

  Step-level updates are stored in a dedicated Redis Hash to avoid the
  N+1 problem of ``load_session → mutate → save_session`` on every step
  change.  The full session snapshot is still available via
  ``save_session`` / ``load_session`` for checkpoint/resume.
  """

  def __init__(self, redis_url: str = "redis://localhost:6379/0"):
    if redis is None:
      raise ImportError("redis package is required for RedisCheckpointStore. Run `pip install redis`.")
    self._redis = redis.from_url(redis_url)

  def _session_key(self, session_id: str) -> str:
    return f"nonoka:checkpoint:session:{session_id}"

  def _steps_key(self, session_id: str) -> str:
    return f"nonoka:checkpoint:steps:{session_id}"

  # ------------------------------------------------------------------ #
  # Full-session snapshot
  # ------------------------------------------------------------------ #

  async def save_session(self, session_id: str, state: SessionState) -> None:
    data = state.model_dump_json()
    await self._redis.set(self._session_key(session_id), data)

  async def load_session(self, session_id: str) -> SessionState | None:
    data = await self._redis.get(self._session_key(session_id))
    if data:
      return SessionState.model_validate_json(data)
    return None

  # ------------------------------------------------------------------ #
  # Granular step updates — Redis Hash (O(1) per field, no load needed)
  # ------------------------------------------------------------------ #

  async def save_step_status(
    self, session_id: str, step_id: str, status: StepStatus
  ) -> None:
    payload = json.dumps({"status": status.value})
    await self._redis.hset(self._steps_key(session_id), step_id, payload)

  async def save_step_result(
    self, session_id: str, step_id: str, result: Any
  ) -> None:
    payload = json.dumps({
      "status": StepStatus.COMPLETED.value,
      "result": result,
    }, default=str)
    await self._redis.hset(self._steps_key(session_id), step_id, payload)

  async def save_step_error(
    self, session_id: str, step_id: str, error: Exception
  ) -> None:
    payload = json.dumps({
      "status": StepStatus.FAILED.value,
      "error": {
        "error_type": type(error).__name__,
        "message": str(error),
        "traceback": _traceback.format_exc() if _traceback else None,
      },
    })
    await self._redis.hset(self._steps_key(session_id), step_id, payload)

  # ------------------------------------------------------------------ #
  # Helpers for loading granular step state back into SessionState
  # ------------------------------------------------------------------ #

  async def load_step_states(self, session_id: str) -> dict[str, dict[str, Any]]:
    """Load all step-level updates for *session_id* as a dict.

    Returns ``{step_id: {"status": ..., "result"|"error": ...}}``.
    """
    raw = await self._redis.hgetall(self._steps_key(session_id))
    if not raw:
      return {}
    parsed: dict[str, dict[str, Any]] = {}
    for step_id_b, payload_b in raw.items():
      step_id = step_id_b.decode("utf-8") if isinstance(step_id_b, bytes) else step_id_b
      payload_str = payload_b.decode("utf-8") if isinstance(payload_b, bytes) else payload_b
      try:
        parsed[step_id] = json.loads(payload_str)
      except json.JSONDecodeError:
        continue
    return parsed

  async def close(self):
    await self._redis.aclose()
