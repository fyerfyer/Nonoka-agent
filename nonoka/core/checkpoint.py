from __future__ import annotations

from typing import Any, Protocol, runtime_checkable
from nonoka.core.session import SessionState, StepStatus

@runtime_checkable
class CheckpointStore(Protocol):
  """
  Persistence protocol.
  """

  # Save/load entire session snapshot
  async def save_session(self, session_id: str, state: SessionState) -> None: ...
  async def load_session(self, session_id: str) -> SessionState | None: ...

  # Granular step updates (avoid rewriting the entire Session every time the Step changes)
  async def save_step_status(self, session_id: str, step_id: str, status: StepStatus) -> None: ...
  async def save_step_result(self, session_id: str, step_id: str, result: Any) -> None: ...
  async def save_step_error(self, session_id: str, step_id: str, error: Exception) -> None: ...