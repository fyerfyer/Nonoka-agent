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
  """

  def __init__(self, redis_url: str = "redis://localhost:6379/0"):
    if redis is None:
      raise ImportError("redis package is required for RedisCheckpointStore. Run `pip install redis`.")
    self._redis = redis.from_url(redis_url)

  def _session_key(self, session_id: str) -> str:
    return f"nonoka:checkpoint:session:{session_id}"

  async def save_session(self, session_id: str, state: SessionState) -> None:
    data = state.model_dump_json()
    await self._redis.set(self._session_key(session_id), data)

  async def load_session(self, session_id: str) -> SessionState | None:
    data = await self._redis.get(self._session_key(session_id))
    if data:
      return SessionState.model_validate_json(data)
    return None

  async def save_step_status(
    self, session_id: str, step_id: str, status: StepStatus
  ) -> None:
    state = await self.load_session(session_id)
    if state:
      state.step_statuses[step_id] = status
      await self.save_session(session_id, state)

  async def save_step_result(
    self, session_id: str, step_id: str, result: Any
  ) -> None:
    state = await self.load_session(session_id)
    if state:
      state.completed_steps[step_id] = StepResult(data=result)
      state.step_statuses[step_id] = StepStatus.COMPLETED
      await self.save_session(session_id, state)

  async def save_step_error(
    self, session_id: str, step_id: str, error: Exception
  ) -> None:
    state = await self.load_session(session_id)
    if state:
      state.failed_steps[step_id] = StepFailure(
        error_type=type(error).__name__,
        message=str(error),
        traceback=_traceback.format_exc() if _traceback else None,
      )
      state.step_statuses[step_id] = StepStatus.FAILED
      await self.save_session(session_id, state)

  async def close(self):
    await self._redis.aclose()
