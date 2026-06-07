from __future__ import annotations

"""Pydantic models for declarative configuration validation."""

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from nonoka.core.types import RetryPolicy


class RetryConfig(BaseModel):
  """Retry policy configuration."""
  max_retries: int = 3
  backoff: float = 2.0

  def to_retry_policy(self) -> RetryPolicy:
    return RetryPolicy(max_retries=self.max_retries, backoff=self.backoff)


class ToolImportConfig(BaseModel):
  """Tool specified by import path."""
  import_: str = Field(..., alias="import")

  @field_validator("import_")
  @classmethod
  def validate_import_path(cls, v: str) -> str:
    if ":" not in v:
      raise ValueError(
        f"Tool import path must be 'module:function_name', got: {v}"
      )
    return v


class AgentConfigModel(BaseModel):
  """Pydantic model for validating a single Agent configuration."""
  model: str | None = None
  system_prompt: str = ""
  tools: list[str | dict[str, Any]] = Field(default_factory=list)
  max_turns: int | None = None
  max_steps: int | None = None
  max_concurrency: int | None = None
  default_retry: RetryConfig | None = None
  default_timeout: float | None = None
  metadata: dict[str, Any] = Field(default_factory=dict)
  tags: list[str] = Field(default_factory=list)

  @model_validator(mode="after")
  def normalize_tools(self) -> AgentConfigModel:
    """Normalize tools to list of import strings."""
    normalized: list[str] = []
    for t in self.tools:
      if isinstance(t, str):
        normalized.append(t)
      elif isinstance(t, dict):
        if "import" in t:
          normalized.append(t["import"])
        else:
          raise ValueError(f"Tool entry must have 'import' key or be a string: {t}")
      else:
        raise ValueError(f"Invalid tool entry type: {type(t)}")
    self.tools = normalized
    return self


class RunnerConfigModel(BaseModel):
  """Pydantic model for validating Runner configuration."""
  checkpoint: str | None = "memory"
  memory: str | None = None


class DefaultsConfigModel(BaseModel):
  """Default values shared across agents."""
  model: str | None = None
  system_prompt: str | None = None
  max_turns: int | None = None
  max_steps: int | None = None
  max_concurrency: int | None = None
  default_retry: RetryConfig | None = None
  default_timeout: float | None = None
  metadata: dict[str, Any] | None = None
  tags: list[str] | None = None


class ConfigFileModel(BaseModel):
  """Top-level configuration file model."""
  agents: dict[str, AgentConfigModel] | None = None
  agent: AgentConfigModel | None = None
  runner: RunnerConfigModel | None = None
  defaults: DefaultsConfigModel | None = None
