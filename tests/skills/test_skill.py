import tempfile
from pathlib import Path

import pytest

from nonoka.skills.skill import Skill


def test_skill_from_string_has_source():
  """from_string should preserve the source identifier."""
  content = """\
---
name: test-skill
description: A test skill
---
This is the activation prompt.
"""
  skill = Skill.from_string(content, source="<test-string>")
  assert skill.name == "test-skill"
  assert skill.description == "A test skill"
  assert skill.source == "<test-string>"


def test_skill_from_file_has_source():
  """from_file should record the absolute path of the skill file."""
  content = """\
---
name: file-skill
description: Loaded from a file
---
Activation prompt.
"""
  with tempfile.TemporaryDirectory() as tmp_dir:
    skill_path = Path(tmp_dir) / "file-skill.md"
    skill_path.write_text(content, encoding="utf-8")

    skill = Skill.from_file(skill_path)
    assert skill.name == "file-skill"
    assert skill.source == str(skill_path.resolve())
