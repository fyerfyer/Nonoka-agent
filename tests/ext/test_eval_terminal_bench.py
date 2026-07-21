"""Fast behavioural coverage for the optional Terminal-Bench adapter."""

from nonoka.ext.eval.terminal_bench import prepare_terminal_session


class _FakeSession:
  def __init__(self) -> None:
    self.calls: list[dict[str, object]] = []

  def send_keys(self, **kwargs: object) -> None:
    self.calls.append(kwargs)


def test_terminal_adapter_disables_interactive_pagers_before_agent_commands():
  session = _FakeSession()

  prepare_terminal_session(session, 42.0)

  assert session.calls == [{
    "keys": ["export PAGER=cat GIT_PAGER=cat GIT_TERMINAL_PROMPT=0 LESS=-FRSX", "Enter"],
    "block": True,
    "max_timeout_sec": 42.0,
  }]
