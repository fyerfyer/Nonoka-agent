from __future__ import annotations

"""Configuration file loading and environment variable substitution."""

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nonoka.core.types import RetryPolicy
from nonoka.config.models import (
  AgentConfigModel,
  ConfigFileModel,
  DefaultsConfigModel,
  RunnerConfigModel,
)
from nonoka.config.resolver import ConfigLoadError, _resolve_tool_entry


# --------------------------------------------------------------------------- #
# Environment variable substitution
# --------------------------------------------------------------------------- #

_ENV_PATTERN = re.compile(r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::-?(?P<default>[^}]*))?\}")


def _substitute_env_vars(value: Any) -> Any:
  """Recursively substitute ``${VAR}`` and ``${VAR:-default}`` in strings."""
  if isinstance(value, str):
    def replacer(match: re.Match[str]) -> str:
      var_name = match.group("name")
      default = match.group("default")
      result = os.getenv(var_name)
      if result is None:
        if default is not None:
          return default
        raise ConfigLoadError(
          f"Environment variable '{var_name}' is not set and no default provided"
        )
      return result
    return _ENV_PATTERN.sub(replacer, value)
  if isinstance(value, dict):
    return {k: _substitute_env_vars(v) for k, v in value.items()}
  if isinstance(value, list):
    return [_substitute_env_vars(item) for item in value]
  return value


# --------------------------------------------------------------------------- #
# Agent / Runner config dataclasses
# --------------------------------------------------------------------------- #

@dataclass
class AgentConfig:
  """Resolved Agent configuration — ready to build."""
  model: str | None = None
  system_prompt: str = ""
  tools: list[str] = field(default_factory=list)  # import paths (lazy resolution)
  max_turns: int | None = None
  max_steps: int | None = None
  max_concurrency: int | None = None
  default_retry: RetryPolicy | None = None
  default_timeout: float | None = None
  metadata: dict[str, Any] = field(default_factory=dict)
  tags: list[str] = field(default_factory=list)

  def build(self) -> Any:
    """Build and return an ``Agent`` instance.

    Tools are resolved from import paths at build time.
    """
    from nonoka.core.agent import Agent

    resolved_tools = [_resolve_tool_entry(path) for path in self.tools]

    kwargs: dict[str, Any] = {
      "tools": resolved_tools,
      "system_prompt": self.system_prompt,
      "metadata": self.metadata,
      "tags": self.tags,
    }
    if self.model is not None:
      kwargs["model"] = self.model
    if self.max_turns is not None:
      kwargs["max_turns"] = self.max_turns
    if self.max_steps is not None:
      kwargs["max_steps"] = self.max_steps
    if self.max_concurrency is not None:
      kwargs["max_concurrency"] = self.max_concurrency
    if self.default_retry is not None:
      kwargs["default_retry"] = self.default_retry
    if self.default_timeout is not None:
      kwargs["default_timeout"] = self.default_timeout

    return Agent(**kwargs)


@dataclass
class RunnerConfig:
  """Resolved Runner configuration — ready to build."""
  checkpoint: str | None = "memory"
  memory: str | None = None

  def build(self) -> Any:
    """Build and return a ``Runner`` instance."""
    from nonoka.core.runner import Runner
    kwargs: dict[str, Any] = {}
    if self.checkpoint is not None:
      kwargs["checkpoint"] = self.checkpoint
    if self.memory is not None:
      kwargs["memory"] = self.memory
    return Runner(**kwargs)


# --------------------------------------------------------------------------- #
# Top-level Config class
# --------------------------------------------------------------------------- #

class Config:
  """Loaded Nonoka configuration.

  Usage::

    config = Config.load("nonoka.yaml")
    agent = config.agents["weather_assistant"].build()
    runner = config.runner.build()
  """

  def __init__(
    self,
    agents: dict[str, AgentConfig] | None = None,
    agent: AgentConfig | None = None,
    runner: RunnerConfig | None = None,
  ):
    self.agents = agents or {}
    self.agent = agent
    self.runner = runner or RunnerConfig()

  @classmethod
  def load(cls, path: str | Path) -> Config:
    """Load configuration from a file path.

    Supports ``.yaml``, ``.yml``, ``.json``, and ``.toml``.
    """
    path = Path(path)
    if not path.exists():
      raise ConfigLoadError(f"Configuration file not found: {path}")

    raw = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()

    if suffix in (".yaml", ".yml"):
      try:
        import yaml
      except ImportError as exc:
        raise ConfigLoadError(
          "PyYAML is required for YAML config. Install: pip install pyyaml"
        ) from exc
      try:
        data = yaml.safe_load(raw)
      except Exception as exc:
        raise ConfigLoadError(f"Failed to parse YAML: {exc}") from exc

    elif suffix == ".json":
      try:
        data = json.loads(raw)
      except json.JSONDecodeError as exc:
        raise ConfigLoadError(f"Failed to parse JSON: {exc}") from exc

    elif suffix == ".toml":
      try:
        if hasattr(__import__("tomllib"), "loads"):
          import tomllib
          data = tomllib.loads(raw)
        else:
          import tomli
          data = tomli.loads(raw)
      except ImportError as exc:
        raise ConfigLoadError(
          "tomli is required for TOML config on Python < 3.11. "
          "Install: pip install tomli"
        ) from exc
      except Exception as exc:
        raise ConfigLoadError(f"Failed to parse TOML: {exc}") from exc
    else:
      raise ConfigLoadError(
        f"Unsupported config format: {suffix}. Use .yaml, .yml, .json, or .toml"
      )

    if data is None:
      data = {}
    if not isinstance(data, dict):
      raise ConfigLoadError(f"Config file must contain a top-level object, got {type(data).__name__}")

    # Substitute env vars before validation
    data = _substitute_env_vars(data)

    try:
      validated = ConfigFileModel.model_validate(data)
    except Exception as exc:
      raise ConfigLoadError(f"Config validation failed: {exc}") from exc

    return cls._from_validated(validated)

  @classmethod
  def auto_find(cls, directory: str | Path | None = None) -> Config:
    """Search for a configuration file in *directory* (default: current working dir).

    Searches for (in order): ``nonoka.yaml``, ``nonoka.yml``, ``nonoka.json``,
    ``nonoka.toml``.
    """
    directory = Path(directory) if directory else Path.cwd()
    candidates = ["nonoka.yaml", "nonoka.yml", "nonoka.json", "nonoka.toml"]
    for name in candidates:
      path = directory / name
      if path.exists():
        return cls.load(path)
    raise ConfigLoadError(
      f"No config file found in {directory}. Searched: {', '.join(candidates)}"
    )

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> Config:
    """Load configuration from a plain dictionary."""
    data = _substitute_env_vars(data)
    try:
      validated = ConfigFileModel.model_validate(data)
    except Exception as exc:
      raise ConfigLoadError(f"Config validation failed: {exc}") from exc
    return cls._from_validated(validated)

  @classmethod
  def _from_validated(cls, validated: ConfigFileModel) -> Config:
    """Build a ``Config`` from a validated ``ConfigFileModel``."""
    defaults = validated.defaults
    runner_cfg = RunnerConfig()
    if validated.runner:
      runner_cfg = RunnerConfig(
        checkpoint=validated.runner.checkpoint,
        memory=validated.runner.memory,
      )

    agents: dict[str, AgentConfig] = {}
    single_agent: AgentConfig | None = None

    def _merge_with_defaults(agent_model: AgentConfigModel) -> AgentConfig:
      return AgentConfig(
        model=agent_model.model or (defaults.model if defaults else None),
        system_prompt=agent_model.system_prompt or (defaults.system_prompt or "") if defaults else agent_model.system_prompt,
        tools=agent_model.tools,
        max_turns=agent_model.max_turns if agent_model.max_turns is not None else (defaults.max_turns if defaults else None),
        max_steps=agent_model.max_steps if agent_model.max_steps is not None else (defaults.max_steps if defaults else None),
        max_concurrency=agent_model.max_concurrency if agent_model.max_concurrency is not None else (defaults.max_concurrency if defaults else None),
        default_retry=(
          agent_model.default_retry.to_retry_policy()
          if agent_model.default_retry
          else (
            defaults.default_retry.to_retry_policy()
            if defaults and defaults.default_retry
            else None
          )
        ),
        default_timeout=agent_model.default_timeout if agent_model.default_timeout is not None else (defaults.default_timeout if defaults else None),
        metadata={**(defaults.metadata if defaults and defaults.metadata else {}), **agent_model.metadata},
        tags=list({*(set(defaults.tags) if defaults and defaults.tags else set()), *agent_model.tags}),
      )

    if validated.agents:
      for name, agent_model in validated.agents.items():
        agents[name] = _merge_with_defaults(agent_model)

    if validated.agent:
      single_agent = _merge_with_defaults(validated.agent)

    return cls(agents=agents, agent=single_agent, runner=runner_cfg)
