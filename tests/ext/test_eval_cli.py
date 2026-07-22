from __future__ import annotations

from types import SimpleNamespace

from nonoka.ext.eval import __main__ as cli
from nonoka.ext.eval.models import EvalSample, StrategyComparison


def test_compare_cli_uses_requested_dataset_and_writes_artifact(monkeypatch, tmp_path, capsys):
  samples = [EvalSample(id="mbpp/1", dataset="mbpp-complex-v1", kind="code", prompt="solve")]
  monkeypatch.setattr(cli, "get_registry", lambda: SimpleNamespace(load=lambda *args: samples))
  captured: dict[str, object] = {}

  async def fake_compare(received, **kwargs):
    captured["samples"] = list(received)
    captured.update(kwargs)
    return StrategyComparison(
      dataset="mbpp-complex-v1", model="test-model", sample_ids=["mbpp/1"], trials=3,
    )

  monkeypatch.setattr(cli, "compare_strategies", fake_compare)
  output = tmp_path / "comparison.json"

  assert cli.main([
    "compare", "--dataset", "mbpp-complex-v1", "--model", "test-model", "--trials", "3",
    "--output", str(output),
  ]) == 0

  assert [sample.id for sample in captured["samples"]] == ["mbpp/1"]
  assert captured["strategies"] == ("direct", "tool_assisted", "verified_repair")
  assert output.is_file()
  assert "Comparison" in capsys.readouterr().out


def test_external_parser_collects_repeatable_agent_kwargs():
  args = cli._build_parser().parse_args([
    "external", "run", "--benchmark", "terminal-bench", "--model", "test",
    "--agent-kwarg", "max_turns=6", "--agent-kwarg", "requires_workspace_mutation=true",
  ])

  assert cli._parse_agent_kwargs(args.agent_kwarg) == {
    "max_turns": 6, "requires_workspace_mutation": True,
  }
