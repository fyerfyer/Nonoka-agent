"""Adapters and prerequisite diagnostics for heavyweight public benchmarks."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
import yaml


@dataclass(frozen=True)
class ExternalBenchmark:
  name: str
  dataset: str
  executable: str
  description: str
  install_hint: str
  requires_docker: bool = True
  python_env_var: str | None = None
  executable_env_var: str | None = None

  def status(self) -> dict[str, Any]:
    interpreter = os.environ.get(self.python_env_var, "") if self.python_env_var else ""
    configured_executable = os.environ.get(self.executable_env_var, "") if self.executable_env_var else ""
    if self.python_env_var:
      available = bool(interpreter and Path(interpreter).is_file())
    elif self.executable_env_var:
      available = bool(configured_executable and Path(configured_executable).is_file())
    else:
      available = shutil.which(self.executable) is not None
    docker = shutil.which("docker") if self.requires_docker else None
    docker_ready = False
    if docker:
      docker_ready = subprocess.run([docker, "info"], capture_output=True, check=False).returncode == 0
    return {
      **asdict(self), "available": available, "docker_ready": docker_ready,
      "interpreter": interpreter or None,
      "configured_executable": configured_executable or None,
    }


TERMINAL_BENCH = ExternalBenchmark(
  name="terminal-bench", dataset="terminal-bench-core==0.1.1", executable="tb",
  description="Real terminal environments with official Docker verifiers.",
  install_hint=(
    "Install terminal-bench and local nonoka in one dedicated Python >=3.12 environment, "
    "then set NONOKA_TERMINAL_BENCH_BIN to that environment's tb executable."
  ),
  executable_env_var="NONOKA_TERMINAL_BENCH_BIN",
)
SWE_BENCH = ExternalBenchmark(
  name="swe-bench", dataset="SWE-bench_Lite", executable="swebench",
  description="Real GitHub issue-to-patch benchmark using Docker.",
  install_hint="Run in a separate host with >=120GB free disk and >=16GB RAM.",
)
TAU2_BENCH = ExternalBenchmark(
  name="tau2-bench", dataset="tau3-bench", executable="tau2",
  description="Official multi-turn customer-service environment with policy and action-level rewards.",
  install_hint=(
    "Create a separate Python 3.12 τ³-bench environment and set "
    "NONOKA_TAU2_PYTHON to its Python executable."
  ),
  requires_docker=False, python_env_var="NONOKA_TAU2_PYTHON",
)
EVALPLUS_BENCH = ExternalBenchmark(
  name="evalplus", dataset="HumanEval+/MBPP+", executable="evalplus.evaluate",
  description="Official strengthened code-generation verifier.",
  install_hint=(
    "Create a separate EvalPlus environment and set NONOKA_EVALPLUS_PYTHON "
    "to its Python executable."
  ),
  requires_docker=False, python_env_var="NONOKA_EVALPLUS_PYTHON",
)
EXTERNAL_BENCHMARKS = (TERMINAL_BENCH, SWE_BENCH, TAU2_BENCH, EVALPLUS_BENCH)


def external_benchmark_status() -> list[dict[str, Any]]:
  return [benchmark.status() for benchmark in EXTERNAL_BENCHMARKS]


def _evaluation_environment() -> dict[str, str]:
  """Build a child-only model configuration environment for external harnesses.

  ``nonoka eval`` is intentionally headless, so it cannot rely on the CLI's
  configuration loader.  Read the same user-owned files without printing any
  credential and keep all values out of command arguments and result artifacts.
  """
  environment = os.environ.copy()
  dotenv_path = Path.home() / ".config" / "nonoka" / ".env"
  for key, value in dotenv_values(dotenv_path).items():
    if value is not None:
      environment.setdefault(key, value)

  config_path = Path(environment.get("NONOKA_EVAL_CONFIG", Path.home() / ".config" / "nonoka" / "config.yaml"))
  if not config_path.is_file():
    return environment
  try:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
  except (OSError, yaml.YAMLError):
    return environment
  if not isinstance(config, dict):
    return environment
  api_key = config.get("api_key")
  base_url = config.get("base_url")
  if isinstance(api_key, str) and api_key:
    environment.setdefault("OPENAI_API_KEY", api_key)
  if isinstance(base_url, str) and base_url:
    environment.setdefault("OPENAI_BASE_URL", base_url)
  return environment


def run_terminal_bench(
  model: str,
  limit: int | None,
  agent: str = "nonoka",
  output: Path | None = None,
  task_ids: list[str] | None = None,
  test_timeout_seconds: float | None = 300.0,
) -> int:
  """Delegate Docker lifecycle and scoring to the official ``tb`` harness."""
  status = TERMINAL_BENCH.status()
  if not status["available"] or not status["docker_ready"]:
    raise RuntimeError("Terminal-Bench unavailable: run `nonoka eval doctor` for remediation.")
  if agent != "nonoka":
    raise ValueError("Only the official Nonoka Terminal-Bench adapter is supported by this command.")
  executable = str(status["configured_executable"] or TERMINAL_BENCH.executable)
  dataset_path = os.environ.get("NONOKA_TERMINAL_BENCH_DATASET_PATH")
  command = [
    executable, "run",
    "--agent-import-path", "nonoka.ext.eval.terminal_bench:NonokaTerminalBenchAgent",
    "--model", model, "--n-concurrent", "1",
  ]
  if dataset_path:
    if not Path(dataset_path).is_dir():
      raise RuntimeError("NONOKA_TERMINAL_BENCH_DATASET_PATH must point to an existing task directory.")
    command[2:2] = ["--dataset-path", dataset_path]
  else:
    command[2:2] = ["--dataset", TERMINAL_BENCH.dataset]
  if task_ids:
    for task_id in task_ids:
      command.extend(["--task-id", task_id])
  elif limit is not None:
    command.extend(["--n-tasks", str(limit)])
  if output is not None:
    output.mkdir(parents=True, exist_ok=True)
    command.extend(["--output-path", str(output)])
  if test_timeout_seconds is not None:
    command.extend(["--global-test-timeout-sec", str(test_timeout_seconds)])
  return subprocess.run(command, env=_evaluation_environment(), check=False).returncode


def run_tau2_bench(
  model: str,
  limit: int | None,
  domain: str,
  max_steps: int,
  timeout: int,
  output: Path,
) -> int:
  """Run Nonoka through the official τ³ benchmark in its isolated environment.

  τ³ pins an older LiteLLM than Nonoka, so the adapter deliberately launches
  it from a separate Python 3.12 environment.  The bridge reads Nonoka's
  normal local configuration; credentials never appear in this command.
  """
  status = TAU2_BENCH.status()
  if not status["available"]:
    raise RuntimeError("τ³-bench unavailable: run `nonoka eval doctor` for remediation.")
  output.mkdir(parents=True, exist_ok=True)
  environment = os.environ.copy()
  environment.setdefault("NONOKA_TAU_BRIDGE_PYTHON", sys.executable)
  environment.setdefault("NONOKA_TAU_EVALUATOR_MODEL", model)
  command = [
    str(status["interpreter"]), str(Path(__file__).with_name("tau2_adapter.py")), "run",
    "--domain", domain,
    "--agent", "nonoka_tau", "--agent-llm", model,
    "--user", "nonoka_tau_user", "--user-llm", model,
    "--num-trials", "1",
    "--max-steps", str(max_steps), "--timeout", str(timeout),
    "--enforce-communication-protocol",
    "--max-concurrency", "1", "--save-to", str(output),
  ]
  if limit is not None:
    command.extend(["--num-tasks", str(limit)])
  return subprocess.run(command, env=environment, check=False).returncode


def export_evalplus_tasks(dataset: str, output: Path) -> None:
  status = EVALPLUS_BENCH.status()
  if not status["available"]:
    raise RuntimeError("EvalPlus unavailable: run `nonoka eval doctor` for remediation.")
  command = [
    str(status["interpreter"]), str(Path(__file__).with_name("evalplus_adapter.py")), "export",
    "--dataset", dataset, "--output", str(output),
  ]
  if subprocess.run(command, check=False).returncode:
    raise RuntimeError("EvalPlus task export failed.")


def run_evalplus(dataset: str, candidates: Path, parallel: int = 1) -> Path:
  status = EVALPLUS_BENCH.status()
  if not status["available"]:
    raise RuntimeError("EvalPlus unavailable: run `nonoka eval doctor` for remediation.")
  command = [
    str(status["interpreter"]), str(Path(__file__).with_name("evalplus_adapter.py")), "evaluate",
    "--dataset", dataset, "--samples", str(candidates), "--parallel", str(parallel),
  ]
  result_path = candidates.with_name(f"{candidates.stem}_eval_results.json")
  # EvalPlus uses a non-zero process status when any candidate fails its
  # strengthened tests.  That is a scored benchmark outcome, not harness
  # failure, provided it emitted the official result artifact.
  returncode = subprocess.run(command, check=False).returncode
  if returncode and not result_path.is_file():
    raise RuntimeError("Official EvalPlus verification failed before producing results.")
  return result_path
