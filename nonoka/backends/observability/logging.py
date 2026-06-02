import logging
from nonoka.core.event import ObservabilityBackend, AgentEvent


class StructuredLoggingBackend(ObservabilityBackend):
  """
  Based on Python standard logging to implement a structured logging backend.
  Since the core layer EventType can be a string or Enum, it is compatible here.
  """

  def __init__(self, logger_name: str = "nonoka.agent"):
    # Get the logger for the framework
    self.logger = logging.getLogger(logger_name)
    # It is recommended that the user configure logging.basicConfig() uniformly when the application is started (or in the CLI)
    # This function is only responsible for outputting in a structured manner, not hardcoding the log format (Formatter)

  async def on_event(self, event: AgentEvent) -> None:
    """Receive AgentEvent and output it through logging"""

    # Handle Enum and plain string event types
    event_type = event.type.value if hasattr(event.type, 'value') else str(event.type)

    log_data = {
      "session_id": event.session_id,
      "event_type": event_type,
      "timestamp": event.timestamp.isoformat(),
      "data": event.data
    }

    log_msg = f"[{event_type}] Session={event.session_id}"

    # Dynamic routing based on the root of the event level
    if "failed" in event_type or "error" in event_type:
      self.logger.error(log_msg, extra={"agent_event": log_data})
    elif "started" in event_type or "completed" in event_type:
      self.logger.info(log_msg, extra={"agent_event": log_data})
    else:
      self.logger.debug(log_msg, extra={"agent_event": log_data})