"""Small official-EvalPlus helper, run in the isolated EvalPlus environment."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _sanitize_proxy() -> None:
  for key in ("ALL_PROXY", "all_proxy"):
    if os.environ.get(key, "").lower().startswith("socks://"):
      os.environ.pop(key, None)


def _import_evalplus() -> None:
  """Avoid shadowing Hugging Face's ``datasets`` with Nonoka's sibling package."""
  script_dir = Path(__file__).resolve().parent
  sys.path[:] = [
    entry for entry in sys.path
    if Path(entry or os.curdir).resolve() != script_dir
  ]


def export_tasks(dataset: str, output: Path) -> int:
  _import_evalplus()
  if dataset == "humaneval":
    from evalplus.data import get_human_eval_plus

    tasks = get_human_eval_plus()
  elif dataset == "mbpp":
    from evalplus.data import get_mbpp_plus

    tasks = get_mbpp_plus()
  else:
    raise ValueError(f"Unsupported EvalPlus dataset: {dataset}")
  output.parent.mkdir(parents=True, exist_ok=True)
  with output.open("w", encoding="utf-8") as handle:
    for task_id, task in tasks.items():
      handle.write(json.dumps({
        "task_id": task_id,
        "prompt": task["prompt"],
        "entry_point": task.get("entry_point"),
      }) + "\n")
  return 0


def evaluate(dataset: str, samples: Path, parallel: int) -> int:
  _import_evalplus()
  from evalplus.evaluate import evaluate as official_evaluate

  official_evaluate(dataset, samples=str(samples), parallel=parallel, i_just_wanna_run=True)
  return 0


def main() -> int:
  parser = argparse.ArgumentParser()
  commands = parser.add_subparsers(dest="command", required=True)
  export = commands.add_parser("export")
  export.add_argument("--dataset", choices=["humaneval", "mbpp"], required=True)
  export.add_argument("--output", type=Path, required=True)
  verify = commands.add_parser("evaluate")
  verify.add_argument("--dataset", choices=["humaneval", "mbpp"], required=True)
  verify.add_argument("--samples", type=Path, required=True)
  verify.add_argument("--parallel", type=int, default=1)
  args = parser.parse_args()
  _sanitize_proxy()
  if args.command == "export":
    return export_tasks(args.dataset, args.output)
  return evaluate(args.dataset, args.samples, args.parallel)


if __name__ == "__main__":
  raise SystemExit(main())
