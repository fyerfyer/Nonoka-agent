"""Fast behavioural coverage for the optional Terminal-Bench adapter."""

import base64

import pytest

from nonoka.ext.eval.terminal_bench import (
  _terminal_submission,
  _validate_terminal_command,
  prepare_terminal_session,
)


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


def test_terminal_adapter_allows_multiline_shell_programs_for_here_documents():
  command = "cat > note.txt <<'EOF'\nhello\nEOF"
  submission = _terminal_submission(command)

  assert "\n" not in submission
  encoded = submission.split("'")[3]
  assert base64.b64decode(encoded).decode("utf-8") == command


def test_terminal_adapter_rejects_nul_bytes():
  with pytest.raises(ValueError, match="NUL"):
    _validate_terminal_command("echo bad\x00input")
