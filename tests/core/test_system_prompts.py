"""Tests for built-in system prompt templates."""

from nonoka.core.system_prompts import SystemPromptTemplate


def test_exploration_contains_has_more_guidance():
  """EXPLORATION template should mention has_more semantics."""
  prompt = SystemPromptTemplate.EXPLORATION
  assert "has_more" in prompt.lower() or "stop" in prompt.lower()


def test_exploration_template_avoids_loop_language():
  """EXPLORATION should explicitly discourage tool-call loops."""
  prompt = SystemPromptTemplate.EXPLORATION
  assert "stop" in prompt.lower()


def test_deterministic_minimizes_tool_calls():
  """DETERMINISTIC should tell the LLM to avoid unnecessary tools."""
  prompt = SystemPromptTemplate.DETERMINISTIC
  assert "only use tools" in prompt.lower() or "minim" in prompt.lower()


def test_coding_emphasizes_structure():
  """CODING should mention type hints / docstrings."""
  prompt = SystemPromptTemplate.CODING
  assert "type hint" in prompt.lower() or "docstring" in prompt.lower()


def test_coding_requires_evidence_preservation_and_acceptance_checks():
  """CODING should preserve fragile inputs and verify observable outcomes."""
  prompt = SystemPromptTemplate.CODING.lower()
  assert "volatile evidence" in prompt
  assert "acceptance checklist" in prompt
  assert "materially different fallback" in prompt


def test_exploration_with_extra_context():
  """exploration() should prepend extra context."""
  extra = "You are an expert researcher."
  prompt = SystemPromptTemplate.exploration(extra)
  assert prompt.startswith(extra)
  assert "has_more" in prompt.lower()


def test_deterministic_with_extra_context():
  """deterministic() should prepend extra context."""
  extra = "Be concise."
  prompt = SystemPromptTemplate.deterministic(extra)
  assert prompt.startswith(extra)
  assert "only use tools" in prompt.lower()


def test_coding_with_extra_context():
  """coding() should prepend extra context."""
  extra = "Focus on asyncio."
  prompt = SystemPromptTemplate.coding(extra)
  assert prompt.startswith(extra)
  assert "type hint" in prompt.lower()
