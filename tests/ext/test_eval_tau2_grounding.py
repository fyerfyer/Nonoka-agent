from nonoka.ext.eval.tau2_grounding import validate_final_response


def test_tau_grounding_rejects_conflicting_available_sku_count():
  messages = [{
    "role": "tool",
    "content": '{"inventory": {"available_skus": 10, "status": "available"}}',
  }]

  finding = validate_final_response(messages, "There are 12 available SKU options.")

  assert finding.passed is False
  assert "12" in finding.feedback
  assert "available_skus" in finding.fact_paths[0]


def test_tau_grounding_accepts_tool_backed_count_and_state():
  messages = [{
    "role": "tool",
    "content": '{"inventory": {"available_skus": 10, "status": "available"}}',
  }]

  finding = validate_final_response(messages, "There are 10 available SKU options.")

  assert finding.passed is True
