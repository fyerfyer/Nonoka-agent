"""Tests for the declarative configuration system."""

import json
import os
import tempfile
from pathlib import Path

import pytest
import yaml

from nonoka.core.agent import Agent
from nonoka.core.builder import AgentBuilder, RunnerBuilder
from nonoka.config import (
  Config,
  ConfigLoadError,
  resolve_tool_import,
  _substitute_env_vars,
)
from nonoka.core.runner import Runner
from nonoka.core.tool import tool
from nonoka.core.types import RetryPolicy


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

@tool
async def dummy_tool(x: str) -> str:
  """A dummy tool for testing."""
  return f"result: {x}"


def write_yaml(path: Path, data: dict) -> None:
  path.write_text(yaml.dump(data), encoding="utf-8")


def write_json(path: Path, data: dict) -> None:
  path.write_text(json.dumps(data), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Environment variable substitution
# --------------------------------------------------------------------------- #

def test_substitute_env_vars_basic():
  os.environ["NONOKA_TEST_VAR"] = "hello"
  assert _substitute_env_vars("${NONOKA_TEST_VAR}") == "hello"
  del os.environ["NONOKA_TEST_VAR"]


def test_substitute_env_vars_with_default():
  assert _substitute_env_vars("${NONOKA_TEST_NONEXIST:-fallback}") == "fallback"


def test_substitute_env_vars_missing_raises():
  with pytest.raises(ConfigLoadError):
    _substitute_env_vars("${NONOKA_TEST_MUST_EXIST}")


def test_substitute_env_vars_nested_dict():
  os.environ["NONOKA_TEST_MODEL"] = "gpt-4o"
  data = {"model": "${NONOKA_TEST_MODEL}", "nested": {"key": "${NONOKA_TEST_MODEL}"}}
  result = _substitute_env_vars(data)
  assert result["model"] == "gpt-4o"
  assert result["nested"]["key"] == "gpt-4o"
  del os.environ["NONOKA_TEST_MODEL"]


# --------------------------------------------------------------------------- #
# Tool import resolution
# --------------------------------------------------------------------------- #

def test_resolve_tool_import_builtin():
  """Resolve a built-in function."""
  obj = resolve_tool_import("os.path:join")
  assert obj is os.path.join


def test_resolve_tool_import_invalid_format():
  with pytest.raises(ConfigLoadError):
    resolve_tool_import("no_colon_here")


def test_resolve_tool_import_missing_module():
  with pytest.raises(ConfigLoadError):
    resolve_tool_import("nonexistent_module_xyz:function")


def test_resolve_tool_import_missing_attr():
  with pytest.raises(ConfigLoadError):
    resolve_tool_import("os:nonexistent_attribute_xyz")


# --------------------------------------------------------------------------- #
# Config.load from YAML
# --------------------------------------------------------------------------- #

def test_config_load_yaml_single_agent():
  with tempfile.TemporaryDirectory() as tmpdir:
    path = Path(tmpdir) / "nonoka.yaml"
    write_yaml(path, {
      "agent": {
        "model": "gpt-4o",
        "system_prompt": "You are helpful.",
        "max_turns": 20,
      },
      "runner": {
        "checkpoint": "memory",
      },
    })

    config = Config.load(path)
    assert config.agent is not None
    assert config.agent.model == "gpt-4o"
    assert config.agent.system_prompt == "You are helpful."
    assert config.agent.max_turns == 20
    assert config.runner.checkpoint == "memory"


def test_config_load_yaml_multi_agents():
  with tempfile.TemporaryDirectory() as tmpdir:
    path = Path(tmpdir) / "nonoka.yaml"
    write_yaml(path, {
      "agents": {
        "assistant_a": {
          "model": "gpt-4o",
          "system_prompt": "A",
        },
        "assistant_b": {
          "model": "deepseek-chat",
          "system_prompt": "B",
          "max_turns": 5,
        },
      },
      "defaults": {
        "model": "default-model",
        "max_turns": 10,
      },
    })

    config = Config.load(path)
    assert "assistant_a" in config.agents
    assert "assistant_b" in config.agents
    # assistant_a inherits defaults
    assert config.agents["assistant_a"].model == "gpt-4o"  # explicit overrides default
    assert config.agents["assistant_a"].max_turns == 10  # from defaults
    # assistant_b has explicit values
    assert config.agents["assistant_b"].model == "deepseek-chat"
    assert config.agents["assistant_b"].max_turns == 5


def test_config_load_yaml_with_retry():
  with tempfile.TemporaryDirectory() as tmpdir:
    path = Path(tmpdir) / "nonoka.yaml"
    write_yaml(path, {
      "agent": {
        "model": "gpt-4o",
        "default_retry": {
          "max_retries": 5,
          "backoff": 1.5,
        },
      },
    })

    config = Config.load(path)
    assert config.agent is not None
    assert config.agent.default_retry is not None
    assert config.agent.default_retry.max_retries == 5
    assert config.agent.default_retry.backoff == 1.5


def test_config_load_yaml_with_tools_import():
  with tempfile.TemporaryDirectory() as tmpdir:
    path = Path(tmpdir) / "nonoka.yaml"
    write_yaml(path, {
      "agent": {
        "model": "gpt-4o",
        "tools": [
          {"import": "tests.core.test_config:dummy_tool"},
        ],
      },
    })

    config = Config.load(path)
    assert config.agent is not None
    agent = config.agent.build()
    assert len(agent.tools) == 1
    assert agent.tools[0].name == "dummy_tool"


def test_config_load_yaml_with_env_substitution():
  os.environ["NONOKA_TEST_MODEL"] = "gpt-4o-mini"
  with tempfile.TemporaryDirectory() as tmpdir:
    path = Path(tmpdir) / "nonoka.yaml"
    write_yaml(path, {
      "agent": {
        "model": "${NONOKA_TEST_MODEL}",
      },
    })

    config = Config.load(path)
    assert config.agent is not None
    assert config.agent.model == "gpt-4o-mini"
  del os.environ["NONOKA_TEST_MODEL"]


def test_config_load_unsupported_format():
  with tempfile.TemporaryDirectory() as tmpdir:
    path = Path(tmpdir) / "config.txt"
    path.write_text("hello", encoding="utf-8")
    with pytest.raises(ConfigLoadError):
      Config.load(path)


def test_config_load_missing_file():
  with pytest.raises(ConfigLoadError):
    Config.load("/nonexistent/path/config.yaml")


# --------------------------------------------------------------------------- #
# Config.load from JSON
# --------------------------------------------------------------------------- #

def test_config_load_json():
  with tempfile.TemporaryDirectory() as tmpdir:
    path = Path(tmpdir) / "nonoka.json"
    write_json(path, {
      "agent": {
        "model": "gpt-4o",
        "system_prompt": "Hello from JSON",
      },
    })

    config = Config.load(path)
    assert config.agent is not None
    assert config.agent.model == "gpt-4o"
    assert config.agent.system_prompt == "Hello from JSON"


# --------------------------------------------------------------------------- #
# Config.auto_find
# --------------------------------------------------------------------------- #

def test_config_auto_find_yaml():
  with tempfile.TemporaryDirectory() as tmpdir:
    path = Path(tmpdir) / "nonoka.yaml"
    write_yaml(path, {"agent": {"model": "gpt-4o"}})
    config = Config.auto_find(tmpdir)
    assert config.agent is not None
    assert config.agent.model == "gpt-4o"


def test_config_auto_find_yml():
  with tempfile.TemporaryDirectory() as tmpdir:
    path = Path(tmpdir) / "nonoka.yml"
    write_yaml(path, {"agent": {"model": "gpt-4o"}})
    config = Config.auto_find(tmpdir)
    assert config.agent is not None


def test_config_auto_find_not_found():
  with tempfile.TemporaryDirectory() as tmpdir:
    with pytest.raises(ConfigLoadError):
      Config.auto_find(tmpdir)


# --------------------------------------------------------------------------- #
# Config.from_dict
# --------------------------------------------------------------------------- #

def test_config_from_dict():
  config = Config.from_dict({
    "agents": {
      "test": {
        "model": "gpt-4o",
        "max_turns": 15,
      },
    },
    "runner": {
      "checkpoint": "memory",
      "memory": "in_memory",
    },
  })
  assert config.agents["test"].model == "gpt-4o"
  assert config.agents["test"].max_turns == 15
  assert config.runner.checkpoint == "memory"
  assert config.runner.memory == "in_memory"


# --------------------------------------------------------------------------- #
# AgentConfig.build
# --------------------------------------------------------------------------- #

def test_agent_config_build():
  config = Config.from_dict({
    "agent": {
      "model": "gpt-4o",
      "system_prompt": "You are helpful.",
      "max_turns": 20,
      "metadata": {"category": "test"},
      "tags": ["test"],
    },
  })
  agent = config.agent.build()
  assert isinstance(agent, Agent)
  assert agent.model == "gpt-4o"
  assert agent.system_prompt == "You are helpful."
  assert agent.max_turns == 20
  assert agent.metadata == {"category": "test"}
  assert agent.tags == ["test"]


# --------------------------------------------------------------------------- #
# RunnerConfig.build
# --------------------------------------------------------------------------- #

def test_runner_config_build():
  config = Config.from_dict({
    "runner": {
      "checkpoint": "memory",
    },
  })
  runner = config.runner.build()
  assert isinstance(runner, Runner)


# --------------------------------------------------------------------------- #
# Agent.from_dict
# --------------------------------------------------------------------------- #

def test_agent_from_dict():
  agent = Agent.from_dict({
    "model": "gpt-4o",
    "system_prompt": "You are helpful.",
    "max_turns": 20,
  })
  assert isinstance(agent, Agent)
  assert agent.model == "gpt-4o"
  assert agent.max_turns == 20


def test_agent_from_dict_with_retry_dict():
  agent = Agent.from_dict({
    "model": "gpt-4o",
    "default_retry": {"max_retries": 5, "backoff": 1.5},
  })
  assert agent.default_retry.max_retries == 5
  assert agent.default_retry.backoff == 1.5


def test_agent_from_dict_with_tool_import():
  agent = Agent.from_dict({
    "model": "gpt-4o",
    "tools": ["tests.core.test_config:dummy_tool"],
  })
  assert len(agent.tools) == 1
  assert agent.tools[0].name == "dummy_tool"


def test_agent_from_dict_with_callable_tool():
  agent = Agent.from_dict({
    "model": "gpt-4o",
    "tools": [dummy_tool],
  })
  assert len(agent.tools) == 1
  assert agent.tools[0].name == "dummy_tool"


# --------------------------------------------------------------------------- #
# Agent.from_yaml
# --------------------------------------------------------------------------- #

def test_agent_from_yaml():
  with tempfile.TemporaryDirectory() as tmpdir:
    path = Path(tmpdir) / "agent.yaml"
    write_yaml(path, {
      "model": "gpt-4o",
      "system_prompt": "From YAML",
    })
    agent = Agent.from_yaml(str(path))
    assert agent.model == "gpt-4o"
    assert agent.system_prompt == "From YAML"


# --------------------------------------------------------------------------- #
# Agent.from_json
# --------------------------------------------------------------------------- #

def test_agent_from_json():
  with tempfile.TemporaryDirectory() as tmpdir:
    path = Path(tmpdir) / "agent.json"
    write_json(path, {
      "model": "gpt-4o",
      "system_prompt": "From JSON",
    })
    agent = Agent.from_json(str(path))
    assert agent.model == "gpt-4o"
    assert agent.system_prompt == "From JSON"


# --------------------------------------------------------------------------- #
# AgentBuilder
# --------------------------------------------------------------------------- #

def test_agent_builder_basic():
  agent = (
    AgentBuilder()
    .model("gpt-4o")
    .system_prompt("You are helpful.")
    .max_turns(20)
    .max_steps(100)
    .max_concurrency(5)
    .retry(max_retries=5, backoff=1.5)
    .timeout(45.0)
    .metadata(category="test")
    .tag("production")
    .build()
  )
  assert agent.model == "gpt-4o"
  assert agent.system_prompt == "You are helpful."
  assert agent.max_turns == 20
  assert agent.max_steps == 100
  assert agent.max_concurrency == 5
  assert agent.default_retry.max_retries == 5
  assert agent.default_retry.backoff == 1.5
  assert agent.default_timeout == 45.0
  assert agent.metadata == {"category": "test"}
  assert agent.tags == ["production"]


def test_agent_builder_with_tool():
  agent = (
    AgentBuilder()
    .model("gpt-4o")
    .tool(dummy_tool)
    .build()
  )
  assert len(agent.tools) == 1
  assert agent.tools[0].name == "dummy_tool"


def test_agent_builder_with_tool_by_import():
  agent = (
    AgentBuilder()
    .model("gpt-4o")
    .tool_by_import("tests.core.test_config:dummy_tool")
    .build()
  )
  assert len(agent.tools) == 1
  assert agent.tools[0].name == "dummy_tool"


def test_agent_builder_missing_model_raises():
  with pytest.raises(ValueError, match="model is required"):
    AgentBuilder().build()


# --------------------------------------------------------------------------- #
# RunnerBuilder
# --------------------------------------------------------------------------- #

def test_runner_builder_basic():
  runner = (
    RunnerBuilder()
    .checkpoint("memory")
    .memory("in_memory")
    .build()
  )
  assert isinstance(runner, Runner)


def test_runner_builder_defaults():
  runner = RunnerBuilder().build()
  assert isinstance(runner, Runner)


# --------------------------------------------------------------------------- #
# Defaults merging
# --------------------------------------------------------------------------- #

def test_defaults_merge_with_agent_overrides():
  config = Config.from_dict({
    "defaults": {
      "model": "default-model",
      "max_turns": 10,
      "max_steps": 50,
      "metadata": {"team": "ai"},
      "tags": ["common"],
    },
    "agents": {
      "custom": {
        "model": "gpt-4o",  # override default
        "max_turns": 20,    # override default
        "metadata": {"project": "x"},  # merged with defaults
        "tags": ["special"],  # union with defaults
      },
    },
  })
  agent_cfg = config.agents["custom"]
  assert agent_cfg.model == "gpt-4o"
  assert agent_cfg.max_turns == 20
  assert agent_cfg.max_steps == 50  # from defaults
  assert agent_cfg.metadata == {"team": "ai", "project": "x"}
  assert set(agent_cfg.tags) == {"common", "special"}


# --------------------------------------------------------------------------- #
# Metadata / tags merging in defaults
# --------------------------------------------------------------------------- #

def test_defaults_metadata_merge_single_agent():
  config = Config.from_dict({
    "defaults": {
      "metadata": {"env": "prod"},
      "tags": ["stable"],
    },
    "agent": {
      "model": "gpt-4o",
      "metadata": {"service": "api"},
      "tags": ["v2"],
    },
  })
  assert config.agent.metadata == {"env": "prod", "service": "api"}
  assert set(config.agent.tags) == {"stable", "v2"}
