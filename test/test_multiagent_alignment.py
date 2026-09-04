from types import SimpleNamespace

import multiagent_graph as graph_module


def test_decision_route_reads_validated_json_field():
    assert graph_module.route_after_decision({
        "decision_result": {"decision": "BUY"}
    }) == "paper_buy"
    assert graph_module.route_after_decision({
        "decision_result": {"decision": "SELL"}
    }) == "paper_sell"
    assert graph_module.route_after_decision({
        "decision_result": {"decision": "HOLD"}
    }) == "no_trade"


def test_decision_node_parses_json_instead_of_routing_on_text_prefix(monkeypatch):
    response = {
        "ticker": "005930.KS", "company": "삼성전자", "decision": "BUY",
        "decision_score": 80, "confidence": "HIGH",
        "stock_state": "HIGH_CONVICTION_BUY", "fundamental_gate": "PASS",
        "entry_urgency": "HIGH", "key_positive_factors": ["alignment"],
        "key_risks": [], "reason": "Signals are aligned.",
    }
    fake_agent = SimpleNamespace(invoke=lambda _: {
        "messages": [SimpleNamespace(text=__import__("json").dumps(response))]
    })
    monkeypatch.setattr(graph_module, "decision_agent", fake_agent)

    result = graph_module.decision_node({
        "ticker": "005930.KS",
        "merged_result": {
            "technical": {}, "fundamental": {}, "news": {}, "flow": {},
        },
    })

    assert result["final_decision"] == "BUY"
    assert result["decision_result"]["decision_score"] == 80


def test_decision_node_fails_closed_on_invalid_output(monkeypatch):
    fake_agent = SimpleNamespace(invoke=lambda _: {
        "messages": [SimpleNamespace(text="BUY because it looks strong")]
    })
    monkeypatch.setattr(graph_module, "decision_agent", fake_agent)

    result = graph_module.decision_node({
        "ticker": "005930.KS", "merged_result": {},
    })

    assert result["final_decision"] == "HOLD"
    assert result["decision_result"]["status"] == "ERROR"
    assert result["decision_result"]["confidence"] == "LOW"


def test_buy_execution_graph_does_not_cycle_back_to_decision():
    edges = {
        (edge.source, edge.target)
        for edge in graph_module.graph.get_graph().edges
    }

    assert ("merge", "decision") in edges
    assert ("decision", "ml_filter") in edges
    assert ("portfolio_guard", "paper_order") in edges
    assert ("portfolio_guard", "decision") not in edges
    assert ("decision", "paper_sell") in edges
