import pytest
import logging
from nonoka.backends.observability.logging import StructuredLoggingBackend
from nonoka.core.event import AgentEvent, EventType


@pytest.mark.asyncio
async def test_structured_logging_backend(caplog):
  """Test that observability events are correctly routed to standard logging"""
  caplog.set_level(logging.INFO)

  backend = StructuredLoggingBackend(logger_name="test.logger")

  # 1. Test Info level event
  event1 = AgentEvent(
    type=EventType.SESSION_STARTED,
    session_id="sess-1",
    data={"agent_id": "test-agent"}
  )
  await backend.on_event(event1)

  assert "session.started" in caplog.text
  assert "sess-1" in caplog.text

  # 2. Test Error level event
  caplog.clear()
  event2 = AgentEvent(
    type=EventType.STEP_FAILED,
    session_id="sess-2",
    data={"error": "Tool execution failed"}
  )
  await backend.on_event(event2)

  # Verify it was logged as ERROR
  error_records = [record for record in caplog.records if record.levelname == 'ERROR']
  assert len(error_records) == 1
  assert "step.failed" in error_records[0].message
  assert "sess-2" in error_records[0].message