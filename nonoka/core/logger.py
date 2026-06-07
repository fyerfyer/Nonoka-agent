from __future__ import annotations

"""Unified logging utilities for nonoka core.

All core modules should import the logger from here rather than
creating their own, ensuring a consistent logging source and
eliminating the need for ``try / except ImportError`` fallback
boiler-plate.
"""

from typing import Any

import structlog


def get_logger(name: str) -> Any:
  """Return a structlog logger for the given dotted module name."""
  return structlog.get_logger(name)
