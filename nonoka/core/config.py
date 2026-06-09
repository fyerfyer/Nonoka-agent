from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class NonokaSettings(BaseSettings):
  """
  The global configuration for the Nonoka framework, supporting reading from environment variables or .env files.
  For example, use NONOKA_DEFAULT_MODEL to override default_model.
  """
  # LLM
  default_model: str = "deepseek-chat"
  openai_api_key: str | None = None
  anthropic_api_key: str | None = None
  openai_base_url: str | None = None 
  anthropic_base_url: str | None = None 

  # Execution policy
  max_steps: int = 50
  max_turns: int = 10
  default_timeout: float = 60.0
  max_concurrency: int = 10

  # Memory backend configuration
  memory_backend: str = "sqlite"        # Optional: sqlite, in_memory

  # Checkpoint backend configuration
  checkpoint_backend: str = "sqlite"    # Optional: sqlite, memory
  checkpoint_interval: str = "per_step" # Optional: per_step, per_layer

  # Observability
  otel_endpoint: str | None = None
  prometheus_port: int | None = None

  model_config = SettingsConfigDict(env_prefix="NONOKA_", env_file=".env", extra="ignore")


# Export singleton instance
settings = NonokaSettings()