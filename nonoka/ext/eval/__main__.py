"""CLI entry point for ``python -m nonoka.ext.eval``."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from nonoka.ext.eval.datasets import get_registry
from nonoka.ext.eval.external import external_benchmark_status, run_tau2_bench, run_terminal_bench
from nonoka.ext.eval.models import EvalRun, Leaderboard
from nonoka.ext.eval.matrix import build_manifest, run_manifest, write_manifest


def _eval_dir() -> Path:
  return Path.cwd() / ".nonoka" / "eval"


def _load_env_files() -> None:
  for path in (Path.home() / ".config" / "nonoka" / ".env", Path.cwd() / ".env"):
    if path.exists():
      load_dotenv(path, override=False)


def cmd_list(_: argparse.Namespace) -> int:
  print("Built-in datasets:")
  for dataset in get_registry().list():
    print(f"  {dataset.name:<10} {dataset.description}\n{' ':14}source: {dataset.source}")
  return 0


async def _run(args: argparse.Namespace) -> EvalRun:
  # Keep listing, local leaderboard reads, and prerequisite checks free from
  # LiteLLM initialization (and therefore free from any model-cost-map network
  # lookups or proxy requirements).
  from nonoka.ext.eval.runners.headless import HeadlessEvalRunner

  samples = get_registry().load(args.dataset, args.limit, args.offset)
  if not samples:
    raise RuntimeError("Dataset is empty")
  runner = HeadlessEvalRunner(
    args.model, max_turns=args.max_turns, timeout_seconds=args.timeout,
    temperature=args.temperature,
  )
  results = await runner.evaluate_many(samples)
  baseline = await runner.evaluate_many(samples, baseline=True) if args.baseline else []
  return EvalRun(
    dataset=args.dataset, model=args.model, samples=results, baseline_samples=baseline,
    metadata={
      "limit": args.limit,
      "offset": args.offset,
      "max_turns": args.max_turns,
      "temperature": args.temperature,
      "baseline": args.baseline,
    },
  )


def _print_summary(run: EvalRun) -> None:
  summary = run.summary()
  print(f"Run {summary['run_id']} completed")
  print(f"  dataset: {summary['dataset']}  model: {summary['model']}")
  print(f"  passed: {summary['passed']}/{summary['total']}  pass@1: {summary['pass_at_1']:.2%}")
  print(f"  avg turns: {summary['avg_turns']:.1f}  llm calls: {summary['avg_llm_calls']:.1f}  tool calls: {summary['avg_tool_calls']:.1f}")
  print(f"  tokens: {summary['total_tokens']}  cost USD: {summary['total_estimated_cost_usd']}")
  if summary["direct_pass_at_1"] is not None:
    print(f"  direct pass@1: {summary['direct_pass_at_1']:.2%}  agent lift: {summary['agent_lift']:+.2%}")


def cmd_run(args: argparse.Namespace) -> int:
  if not args.model:
    print("Error: --model is required.", file=sys.stderr)
    return 2
  try:
    run = asyncio.run(_run(args))
  except Exception as exc:
    print(f"Error: {exc}", file=sys.stderr)
    return 1
  output = Path(args.output) if args.output else _eval_dir() / "runs" / f"{run.run_id}.json"
  run.write(output)
  board_path = _eval_dir() / "leaderboard.json"
  board = Leaderboard.load(board_path)
  board.append(run)
  board.save(board_path)
  _print_summary(run)
  print(f"Results written to {output}")
  if args.json:
    print(run.model_dump_json(indent=2))
  return 0


def cmd_leaderboard(args: argparse.Namespace) -> int:
  entries = Leaderboard.load(_eval_dir() / "leaderboard.json").entries
  entries = [entry for entry in entries if (not args.dataset or entry["dataset"] == args.dataset) and (not args.model or entry["model"] == args.model)]
  if not entries:
    print("No matching leaderboard data yet.")
    return 0
  print(f"{'dataset':<12} {'model':<24} {'pass@1':>8} {'turns':>7} {'tokens':>9} {'run':<12}")
  for entry in sorted(entries, key=lambda item: item["pass_at_1"], reverse=True):
    print(f"{entry['dataset']:<12} {entry['model']:<24.24} {entry['pass_at_1']:>7.1%} {entry['avg_turns']:>7.1f} {entry['total_tokens']:>9} {entry['run_id']:<12}")
  return 0


def cmd_doctor(args: argparse.Namespace) -> int:
  payload = {"python": sys.version.split()[0], "external_benchmarks": external_benchmark_status()}
  if args.json:
    print(json.dumps(payload, indent=2))
  else:
    print(f"Python {payload['python']}")
    for benchmark in payload["external_benchmarks"]:
      ready = benchmark["available"] and (not benchmark["requires_docker"] or benchmark["docker_ready"])
      icon = "OK" if ready else "WARN"
      print(f"{icon:4} {benchmark['name']}: executable={benchmark['available']} docker={benchmark['docker_ready']}")
      if icon == "WARN":
        print(f"     {benchmark['install_hint']}")
  return 0


def cmd_external_run(args: argparse.Namespace) -> int:
  try:
    if args.benchmark == "terminal-bench":
      output = Path(args.output) if args.output else _eval_dir() / "external" / "terminal-bench"
      return run_terminal_bench(
        args.model, args.limit, args.agent, output, args.task_ids, args.test_timeout,
      )
    if args.benchmark == "tau2-bench":
      output = Path(args.output) if args.output else _eval_dir() / "external" / "tau2-bench"
      return run_tau2_bench(args.model, args.limit, args.domain, args.max_steps, args.timeout, output)
    print("SWE-bench remains documented as an external harness due to its host resource requirements.", file=sys.stderr)
    return 2
  except Exception as exc:
    print(f"Error: {exc}", file=sys.stderr)
    return 1


def cmd_matrix_plan(args: argparse.Namespace) -> int:
  path = Path(args.output) if args.output else _eval_dir() / "matrix" / "manifest.json"
  manifest = build_manifest(args.model, args.temperature, args.max_turns, args.timeout)
  write_manifest(manifest, path)
  print(f"Matrix manifest written to {path}")
  for job in manifest["jobs"]:
    print(f"  {job['id']:<20} {job['runner']}")
  return 0


def cmd_matrix_run(args: argparse.Namespace) -> int:
  try:
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    selected = set(args.include) if args.include else None
    output = Path(args.output) if args.output else Path(args.manifest).with_suffix("")
    result = asyncio.run(run_manifest(manifest, output, selected))
  except Exception as exc:
    print(f"Error: {exc}", file=sys.stderr)
    return 1
  for record in result["records"]:
    print(f"{record['status']:<10} {record['id']}")
  return 0


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(prog="python -m nonoka.ext.eval", description="Nonoka reproducible agent evaluation")
  subparsers = parser.add_subparsers(dest="command")
  list_parser = subparsers.add_parser("list", help="List built-in datasets")
  list_parser.set_defaults(func=cmd_list)
  run_parser = subparsers.add_parser("run", help="Run a framework-owned benchmark")
  run_parser.add_argument("--dataset", "-d", required=True)
  run_parser.add_argument("--model", required=True)
  run_parser.add_argument("--limit", "-n", type=int, default=None)
  run_parser.add_argument("--offset", type=int, default=0, help="Skip this many samples before applying --limit")
  run_parser.add_argument("--max-turns", type=int, default=8)
  run_parser.add_argument("--timeout", type=float, default=90.0)
  run_parser.add_argument("--temperature", type=float, default=0.0)
  run_parser.add_argument("--no-baseline", dest="baseline", action="store_false", help="Skip direct paired baseline")
  run_parser.set_defaults(baseline=True)
  run_parser.add_argument("--output")
  run_parser.add_argument("--json", action="store_true")
  run_parser.set_defaults(func=cmd_run)
  board_parser = subparsers.add_parser("leaderboard", help="Show local leaderboard")
  board_parser.add_argument("--dataset")
  board_parser.add_argument("--model")
  board_parser.set_defaults(func=cmd_leaderboard)
  doctor_parser = subparsers.add_parser("doctor", help="Check Docker benchmark prerequisites")
  doctor_parser.add_argument("--json", action="store_true")
  doctor_parser.set_defaults(func=cmd_doctor)
  external_parser = subparsers.add_parser("external", help="Run external official benchmark harnesses")
  external_subparsers = external_parser.add_subparsers(dest="external_command", required=True)
  external_run = external_subparsers.add_parser("run")
  external_run.add_argument("--benchmark", required=True, choices=["terminal-bench", "tau2-bench", "swe-bench"])
  external_run.add_argument("--model", required=True)
  external_run.add_argument("--limit", type=int, default=10)
  external_run.add_argument("--task-id", dest="task_ids", action="append", help="Terminal-Bench task id; repeat to select multiple tasks.")
  external_run.add_argument("--test-timeout", type=float, default=300.0, help="Official Terminal-Bench verifier timeout in seconds.")
  external_run.add_argument("--agent", default=os.environ.get("NONOKA_HARBOR_AGENT", "nonoka"))
  external_run.add_argument("--domain", default="retail", choices=["retail", "airline", "telecom", "telecom-workflow", "banking_knowledge"])
  external_run.add_argument("--max-steps", type=int, default=24)
  external_run.add_argument("--timeout", type=int, default=300)
  external_run.add_argument("--output", help="Directory for the official external harness artifact.")
  external_run.set_defaults(func=cmd_external_run)
  matrix_parser = subparsers.add_parser("matrix", help="Plan or execute a reproducible release benchmark matrix")
  matrix_subparsers = matrix_parser.add_subparsers(dest="matrix_command", required=True)
  matrix_plan = matrix_subparsers.add_parser("plan", help="Write a pinned benchmark manifest without model calls")
  matrix_plan.add_argument("--model", required=True)
  matrix_plan.add_argument("--temperature", type=float, default=0.0)
  matrix_plan.add_argument("--max-turns", type=int, default=8)
  matrix_plan.add_argument("--timeout", type=float, default=90.0)
  matrix_plan.add_argument("--output")
  matrix_plan.set_defaults(func=cmd_matrix_plan)
  matrix_run = matrix_subparsers.add_parser("run", help="Run selected jobs from a matrix manifest")
  matrix_run.add_argument("--manifest", required=True)
  matrix_run.add_argument("--include", nargs="+")
  matrix_run.add_argument("--output")
  matrix_run.set_defaults(func=cmd_matrix_run)
  return parser


def main(argv: list[str] | None = None) -> int:
  _load_env_files()
  parser = _build_parser()
  args = parser.parse_args(argv)
  if not hasattr(args, "func"):
    parser.print_help()
    return 2
  return int(args.func(args))


if __name__ == "__main__":
  raise SystemExit(main())
