from __future__ import annotations

from typing import Any
from nonoka.core.checkpoint import CheckpointStore
from nonoka.core.session import SessionState, StepStatus


class NoOpCheckpointStore(CheckpointStore):
  """
  No-op checkpoint store that does not persist anything.

  Useful when persistence is explicitly disabled::

    runner = Runner(checkpoint="disabled")
  """

  async def delete_session(self, session_id: str) -> bool:
    return False

  async def save_session(self, session_id: str, state: SessionState) -> None:
    pass

  async def load_session(self, session_id: str) -> SessionState | None:
    return None

  async def save_step_status(
    self, session_id: str, step_id: str, status: StepStatus
  ) -> None:
    pass

  async def save_step_result(
    self, session_id: str, step_id: str, result: Any
  ) -> None:
    pass

  async def save_step_error(
    self, session_id: str, step_id: str, error: Exception
  ) -> None:
    pass
