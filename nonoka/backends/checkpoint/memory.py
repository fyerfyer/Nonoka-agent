from __future__ import annotations

import asyncio
from typing import Any
import traceback as _traceback

from nonoka.core.checkpoint import CheckpointStore
from nonoka.core.session import SessionState, StepStatus, StepResult, StepFailure


class MemoryCheckpointStore(CheckpointStore):
  """
  In-memory checkpoint store for development and testing.
  """

  def __init__(self):
    self._sessions: dict[str, SessionState] = {}
    self._lock = asyncio.Lock()

  async def delete_session(self, session_id: str) -> bool:
    async with self._lock:
      if session_id in self._sessions:
        del self._sessions[session_id]
        return True
      return False

  async def save_session(self, session_id: str, state: SessionState) -> None:
    async with self._lock:
      self._sessions[session_id] = state.model_copy(deep=True)

  async def load_session(self, session_id: str) -> SessionState | None:
    async with self._lock:
      state = self._sessions.get(session_id)
      return state.model_copy(deep=True) if state else None

  async def save_step_status(
    self, session_id: str, step_id: str, status: StepStatus
  ) -> None:
    async with self._lock:
      state = self._sessions.get(session_id)
      if state:
        state.step_statuses[step_id] = status

  async def save_step_result(
    self, session_id: str, step_id: str, result: Any
  ) -> None:
    async with self._lock:
      state = self._sessions.get(session_id)
      if state:
        state.completed_steps[step_id] = StepResult(data=result)
        state.step_statuses[step_id] = StepStatus.COMPLETED

  async def save_step_error(
    self, session_id: str, step_id: str, error: Exception
  ) -> None:
    async with self._lock:
      state = self._sessions.get(session_id)
      if state:
        state.failed_steps[step_id] = StepFailure(
          error_type=type(error).__name__,
          message=str(error),
          traceback=_traceback.format_exc() if _traceback else None,
        )
        state.step_statuses[step_id] = StepStatus.FAILED
