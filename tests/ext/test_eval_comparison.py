from nonoka.ext.eval.models import EvalResult, EvalRun, Metrics, StrategyComparison


def _run(strategy: str, passed: int, tokens: int) -> EvalRun:
  return EvalRun(
    dataset="mbpp", model="fake",
    samples=[
      EvalResult(sample_id="one", success=bool(passed), metrics=Metrics(input_tokens=tokens)),
      EvalResult(sample_id="two", success=False, metrics=Metrics(output_tokens=tokens)),
    ],
    metadata={"strategy": strategy},
  )


def test_strategy_comparison_reports_variance_and_tokens_per_success():
  comparison = StrategyComparison(
    dataset="mbpp", model="fake", sample_ids=["one", "two"], trials=2,
    runs=[_run("direct", 1, 10), _run("direct", 0, 20)],
  )

  summary = comparison.summary()["strategies"]["direct"]

  assert summary["mean_pass_at_1"] == 0.25
  assert summary["pass_rate_variance"] == 0.0625
  assert summary["successes"] == 1
  assert summary["tokens_per_success"] == 60.0
