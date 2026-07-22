from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from nonoka.ext.eval import external


def test_tau2_runner_uses_isolated_interpreter_and_keeps_credentials_out_of_argv(
  monkeypatch, tmp_path,
):
  tau_python = tmp_path / "tau2-python"
  tau_python.touch()
  monkeypatch.setenv("NONOKA_TAU2_PYTHON", str(tau_python))
  captured: dict[str, object] = {}

  class Completed:
    returncode = 0

  def fake_run(command, **kwargs):
    captured["command"] = command
    captured["env"] = kwargs["env"]
    return Completed()

  monkeypatch.setattr(external.subprocess, "run", fake_run)

  output = tmp_path / "results"
  assert external.run_tau2_bench("deepseek-chat", 3, "retail", 24, 300, output) == 0

  command = captured["command"]
  assert command[:3] == [str(tau_python), str(Path(external.__file__).with_name("tau2_adapter.py")), "run"]
  assert ["--agent", "nonoka_tau"] == command[command.index("--agent"):command.index("--agent") + 2]
  assert ["--user", "nonoka_tau_user"] == command[command.index("--user"):command.index("--user") + 2]
  assert "--enforce-communication-protocol" in command
  assert "DEEPSEEK_API_KEY" not in " ".join(command)
  assert captured["env"]["NONOKA_TAU_BRIDGE_PYTHON"] == sys.executable
  assert captured["env"]["NONOKA_TAU_EVALUATOR_MODEL"] == "deepseek-chat"
  assert output.is_dir()


def test_terminal_bench_task_ids_take_precedence_over_limit(monkeypatch, tmp_path):
  executable = tmp_path / "harbor"
  executable.touch()
  monkeypatch.setenv("NONOKA_HARBOR_BIN", str(executable))
  commands: list[list[str]] = []

  class Completed:
    returncode = 0

  def fake_run(command, **kwargs):
    commands.append(command)
    return Completed()

  monkeypatch.setattr(external.shutil, "which", lambda command: "/usr/bin/docker" if command == "docker" else None)
  monkeypatch.setattr(external.subprocess, "run", fake_run)

  external.run_terminal_bench("deepseek-chat", 10, task_ids=["sanitize-git-repo", "configure-git-webserver"])

  command = commands[-1]
  assert "--n-tasks" not in command
  assert command.count("--include-task-name") == 2


def test_terminal_bench_legacy_allows_pinned_local_dataset_path(monkeypatch, tmp_path):
  executable = tmp_path / "tb"
  executable.touch()
  dataset = tmp_path / "tasks"
  dataset.mkdir()
  monkeypatch.setenv("NONOKA_TERMINAL_BENCH_BIN", str(executable))
  monkeypatch.setenv("NONOKA_TERMINAL_BENCH_DATASET_PATH", str(dataset))
  commands: list[list[str]] = []

  class Completed:
    returncode = 0

  def fake_run(command, **kwargs):
    commands.append(command)
    return Completed()

  monkeypatch.setattr(external.shutil, "which", lambda command: "/usr/bin/docker" if command == "docker" else None)
  monkeypatch.setattr(external.subprocess, "run", fake_run)

  external.run_terminal_bench_legacy("deepseek-chat", 1)

  command = commands[-1]
  assert ["--dataset-path", str(dataset)] == command[2:4]
  assert "--dataset" not in command


def test_tau2_status_does_not_require_docker(monkeypatch, tmp_path):
  tau_python = tmp_path / "tau2-python"
  tau_python.touch()
  monkeypatch.setenv("NONOKA_TAU2_PYTHON", str(tau_python))
  monkeypatch.setattr(external.shutil, "which", lambda _: None)

  status = external.TAU2_BENCH.status()

  assert status["available"] is True
  assert status["docker_ready"] is False
  assert status["requires_docker"] is False


def test_terminal_bench_uses_harbor_tb2_adapter(monkeypatch, tmp_path):
  executable = tmp_path / "harbor"
  executable.touch()
  monkeypatch.setenv("NONOKA_HARBOR_BIN", str(executable))
  captured: dict[str, object] = {}

  class Completed:
    returncode = 0

  def fake_run(command, **kwargs):
    captured["command"] = command
    if "env" in kwargs:
      captured["environment"] = kwargs["env"]
    return Completed()

  monkeypatch.setattr(external.shutil, "which", lambda command: "/usr/bin/docker" if command == "docker" else None)
  monkeypatch.setattr(external.subprocess, "run", fake_run)

  output = tmp_path / "run"
  assert external.run_terminal_bench("deepseek-chat", 2, output=output) == 0

  command = captured["command"]
  assert command[:5] == [str(executable), "run", "--dataset", "terminal-bench@2.0", "--agent"]
  assert "nonoka.ext.eval.harbor:NonokaHarborAgent" in command
  assert ["--model", "deepseek-chat"] == command[command.index("--model"):command.index("--model") + 2]
  assert ["--n-tasks", "2"] == command[command.index("--n-tasks"):command.index("--n-tasks") + 2]
  assert ["--jobs-dir", str(output)] == command[command.index("--jobs-dir"):command.index("--jobs-dir") + 2]
  assert "--yes" in command
  assert output.is_dir()
  manifest = (output / "harbor-launch.json").read_text(encoding="utf-8")
  assert "terminal-bench@2.0" in manifest
  assert "--output-path" not in command
  assert "DEEPSEEK_API_KEY" not in " ".join(command)
  assert "--global-test-timeout-sec" not in command


def test_terminal_bench_forwards_json_agent_kwargs(monkeypatch, tmp_path):
  executable = tmp_path / "harbor"
  executable.touch()
  monkeypatch.setenv("NONOKA_HARBOR_BIN", str(executable))
  commands: list[list[str]] = []

  monkeypatch.setattr(external.shutil, "which", lambda command: "/usr/bin/docker" if command == "docker" else None)
  monkeypatch.setattr(external.subprocess, "run", lambda command, **_kwargs: commands.append(command) or SimpleNamespace(returncode=0))

  external.run_terminal_bench("test", 1, agent_kwargs={"max_turns": 6, "requires_workspace_mutation": True})

  command = commands[-1]
  assert ["--agent-kwarg", "max_turns=6"] == command[command.index("--agent-kwarg"):command.index("--agent-kwarg") + 2]
  second = command.index("--agent-kwarg", command.index("--agent-kwarg") + 1)
  assert command[second:second + 2] == ["--agent-kwarg", "requires_workspace_mutation=true"]


def test_external_runner_reads_config_into_child_environment(monkeypatch, tmp_path):
  config = tmp_path / "config.yaml"
  config.write_text("api_key: test-key\nbase_url: https://example.invalid/v1\n")
  monkeypatch.setenv("NONOKA_EVAL_CONFIG", str(config))
  monkeypatch.delenv("OPENAI_API_KEY", raising=False)
  monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

  environment = external._evaluation_environment()

  assert environment["OPENAI_API_KEY"] == "test-key"
  assert environment["OPENAI_BASE_URL"] == "https://example.invalid/v1"


def test_external_runner_removes_only_incompatible_socks_all_proxy(monkeypatch):
  monkeypatch.setenv("ALL_PROXY", "socks://127.0.0.1:7890")
  monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7890")

  environment = external._evaluation_environment()

  assert "ALL_PROXY" not in environment
  assert environment["HTTPS_PROXY"] == "http://127.0.0.1:7890"


def test_evalplus_keeps_scored_result_when_verifier_returns_nonzero(monkeypatch, tmp_path):
  interpreter = tmp_path / "evalplus-python"
  interpreter.touch()
  candidates = tmp_path / "candidates.jsonl"
  candidates.write_text('{"task_id": "Mbpp/2", "solution": "pass"}\n')
  result_path = tmp_path / "candidates_eval_results.json"

  monkeypatch.setattr(
    external, "EVALPLUS_BENCH",
    SimpleNamespace(status=lambda: {"available": True, "interpreter": str(interpreter)}),
  )

  def fake_run(_command, **_kwargs):
    result_path.write_text('{"eval": {}}\n')
    return SimpleNamespace(returncode=1)

  monkeypatch.setattr(external.subprocess, "run", fake_run)

  assert external.run_evalplus("mbpp", candidates) == result_path
