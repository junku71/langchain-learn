import pytest
from pydantic import ValidationError

from analysis.agent_contracts import (
    DecisionResult,
    TechnicalResult,
    specialist_fallback,
    validate_agent_result,
)


def test_json_fence_is_parsed_and_validated():
    result = validate_agent_result(
        TechnicalResult,
        '```json\n{"ticker":"005930.KS","technical_score":77,'
        '"signal":"BULLISH","confidence":"HIGH"}\n```',
    )

    assert result["technical_score"] == 77
    assert result["signal"] == "BULLISH"


def test_decision_contract_rejects_invalid_decision():
    with pytest.raises(ValidationError):
        validate_agent_result(DecisionResult, {
            "ticker": "005930.KS", "decision": "STRONG_BUY",
            "decision_score": 90, "confidence": "HIGH",
            "stock_state": "HIGH_CONVICTION_BUY", "fundamental_gate": "PASS",
            "entry_urgency": "HIGH", "reason": "test",
        })


def test_specialist_failure_is_explicit_low_confidence_fallback():
    result = specialist_fallback("flow", "005930.KS", RuntimeError("offline"))

    assert result["status"] == "ERROR"
    assert result["confidence"] == "LOW"
    assert result["flow_score"] == 50
    assert "offline" in result["error"]
