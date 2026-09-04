"""Validated contracts shared by the stock-analysis LangGraph agents."""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


Confidence = Literal["HIGH", "MEDIUM", "LOW"]
Signal = Literal["BULLISH", "NEUTRAL", "BEARISH"]


class SpecialistResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    ticker: str
    signal: Signal = "NEUTRAL"
    confidence: Confidence = "LOW"
    status: Literal["OK", "ERROR"] = "OK"
    error: str | None = None


class TechnicalResult(SpecialistResult):
    technical_score: float = Field(default=50, ge=0, le=100)
    technical_state: str = "UNKNOWN"
    component_scores: dict[str, float | None] = Field(default_factory=dict)
    key_positive_signals: list[str] = Field(default_factory=list)
    key_warning_signals: list[str] = Field(default_factory=list)
    missing_indicators: list[str] = Field(default_factory=list)
    conclusion: str = ""


class FundamentalResult(SpecialistResult):
    fundamental_score: float = Field(default=50, ge=0, le=100)
    fundamental_state: str = "UNKNOWN"
    component_scores: dict[str, float | None] = Field(default_factory=dict)
    data_coverage_pct: float = Field(default=0, ge=0, le=100)
    key_positive_factors: list[str] = Field(default_factory=list)
    key_risks: list[str] = Field(default_factory=list)
    summary: str = ""


class NewsResult(SpecialistResult):
    news_score: float = Field(default=50, ge=0, le=100)
    sentiment: str = "NEUTRAL"
    catalyst_score: float = Field(default=50, ge=0, le=100)
    earnings_score: float = Field(default=50, ge=0, le=100)
    key_risk: str = ""
    summary: str = ""


class FlowResult(SpecialistResult):
    flow_score: float = Field(default=50, ge=0, le=100)
    flow_state: str = "NEUTRAL"
    component_scores: dict[str, float | None] = Field(default_factory=dict)
    key_positive_signals: list[str] = Field(default_factory=list)
    key_warning_signals: list[str] = Field(default_factory=list)
    summary: str = ""


class DecisionResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    ticker: str
    company: str = ""
    decision: Literal["BUY", "HOLD", "SELL"]
    decision_score: float = Field(ge=0, le=100)
    confidence: Confidence
    stock_state: str
    fundamental_gate: Literal["PASS", "CONDITIONAL_PASS", "FAIL"]
    entry_urgency: Literal["HIGH", "MEDIUM", "LOW"]
    key_positive_factors: list[str] = Field(default_factory=list)
    key_risks: list[str] = Field(default_factory=list)
    reason: str
    status: Literal["OK", "ERROR"] = "OK"
    error: str | None = None


def extract_json(value: Any) -> dict:
    """Extract one JSON object from an agent message without using eval."""
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, dict):
        return value
    text = str(value).strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < start:
            raise ValueError("Agent response does not contain a JSON object")
        text = text[start:end + 1]
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Agent JSON response must be an object")
    return parsed


def validate_agent_result(model: type[BaseModel], value: Any) -> dict:
    return model.model_validate(extract_json(value)).model_dump()


def specialist_fallback(kind: str, ticker: str, error: Exception) -> dict:
    score_key = f"{kind}_score"
    return {
        "ticker": ticker,
        score_key: 50.0,
        "signal": "NEUTRAL",
        "sentiment": "NEUTRAL" if kind == "news" else None,
        "confidence": "LOW",
        "status": "ERROR",
        "error": f"{type(error).__name__}: {error}",
        "summary": "Analysis unavailable; neutral fallback applied.",
    }


__all__ = [
    "DecisionResult", "FlowResult", "FundamentalResult", "NewsResult",
    "TechnicalResult", "ValidationError", "specialist_fallback",
    "validate_agent_result",
]
