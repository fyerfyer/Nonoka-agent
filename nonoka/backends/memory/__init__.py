from __future__ import annotations

from .in_memory import InMemoryBackend

try:
  from .mem0_ext import Mem0Backend
except ImportError:
  Mem0Backend = None

__all__ = ["InMemoryBackend", "Mem0Backend"]
