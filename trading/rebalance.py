from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Literal

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from analysis.news_naver import NaverNewsProvider
from trading.display import stock_name


class RebalanceAction(BaseModel):
    ticker: str
    action: Literal["BUY", "SELL", "HOLD", "REDUCE"]
    target_weight_pct: float = Field(ge=0, le=100)
    confidence: float = Field(ge=0, le=1)
    reason: str
    supporting_factors: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)


class RebalanceProposal(BaseModel):
    market_view: Literal["RISK_ON", "NEUTRAL", "RISK_OFF"]
    market_summary: str
    recommended_cash_pct: float = Field(ge=0, le=100)
    actions: list[RebalanceAction]
    overall_reason: str
    portfolio_assessment: str = ""
    news_assessment: str = ""
    investment_thesis: str = ""
    risk_scenarios: list[str] = Field(default_factory=list)
    implementation_notes: list[str] = Field(default_factory=list)


class MarketNewsService:
    MARKET_NEWS_MAX_AGE = timedelta(days=7)
    QUERIES = (
        "오늘 코스피 코스닥 증시 외국인 기관 환율",
        "오늘 한국 주식시장 업종 반도체 자동차 바이오",
        "미국 증시 금리 달러 유가 한국 증시 영향",
    )
    POSITIVE = ("상승", "강세", "순매수", "회복", "개선", "호재")
    NEGATIVE = ("하락", "약세", "순매도", "우려", "악화", "충격")

    def __init__(self, provider: NaverNewsProvider | None = None):
        self.provider = provider or NaverNewsProvider()

    @staticmethod
    def _published_at(value) -> datetime | None:
        """Normalize Naver/RSS dates and Unix timestamps to an aware datetime."""
        if value in (None, ""):
            return None
        try:
            if isinstance(value, (int, float)):
                timestamp = float(value)
                if timestamp > 10_000_000_000:
                    timestamp /= 1000
                return datetime.fromtimestamp(timestamp, tz=timezone.utc)
            parsed = parsedate_to_datetime(str(value))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except (TypeError, ValueError, OverflowError):
            try:
                parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return parsed.astimezone(timezone.utc)
            except (TypeError, ValueError):
                return None

    @staticmethod
    def _security_news_summary(headlines: list[dict]) -> str:
        """Return up to three short, link-free lines about the security."""
        lines = []
        for item in headlines[:3]:
            title = " ".join(str(item.get("title") or "").split())
            description = " ".join(str(item.get("description") or "").split())
            detail = description if description and description not in title else ""
            line = f"{title} — {detail}" if detail else title
            if line:
                lines.append(line[:180].rstrip())
        return "\n".join(lines) or "최근 확인된 종목 관련 뉴스가 없습니다."

    def collect(self) -> dict:
        items: list[dict] = []
        errors: list[str] = []
        collected_at = datetime.now().astimezone()
        cutoff = collected_at.astimezone(timezone.utc) - self.MARKET_NEWS_MAX_AGE
        for query in self.QUERIES:
            try:
                for item in self.provider.search_query(query, display=10):
                    row = item.to_dict()
                    published_at = self._published_at(row.get("published_at"))
                    if published_at is None or published_at < cutoff:
                        continue
                    row["query"] = query
                    items.append(row)
            except Exception as error:
                errors.append(f"{type(error).__name__}: {error}")
        unique = {item.get("link") or item.get("title"): item for item in items}
        headlines = list(unique.values())[:25]
        text = " ".join(
            f"{item.get('title', '')} {item.get('description', '')}"
            for item in headlines
        )
        positive = sum(text.count(word) for word in self.POSITIVE)
        negative = sum(text.count(word) for word in self.NEGATIVE)
        sentiment = "POSITIVE" if positive > negative else (
            "NEGATIVE" if negative > positive else "NEUTRAL"
        )
        return {
            "collected_at": collected_at.isoformat(),
            "news_since": cutoff.isoformat(),
            "sentiment": sentiment,
            "positive_hits": positive,
            "negative_hits": negative,
            "headlines": headlines,
            "errors": errors,
        }

    def collect_securities(self, securities: list[dict], display: int = 5) -> dict:
        """Collect deduplicated company news for held and recommended securities."""
        results = {}
        seen = set()
        for security in securities:
            ticker = str(security.get("ticker") or "").upper()
            if not ticker or ticker in seen:
                continue
            seen.add(ticker)
            name = str(security.get("name") or stock_name(ticker))
            try:
                headlines = [
                    item.to_dict()
                    for item in self.provider.search_query(
                        f"{name} 주가 실적 전망", display=display
                    )
                ]
                text = " ".join(
                    f"{item.get('title', '')} {item.get('description', '')}"
                    for item in headlines
                )
                positive = sum(text.count(word) for word in self.POSITIVE)
                negative = sum(text.count(word) for word in self.NEGATIVE)
                sentiment = "POSITIVE" if positive > negative else (
                    "NEGATIVE" if negative > positive else "NEUTRAL"
                )
                summary = self._security_news_summary(headlines)
                error = ""
            except Exception as exc:
                headlines, sentiment = [], "UNAVAILABLE"
                positive = negative = 0
                summary = "종목 뉴스를 수집하지 못해 뉴스 판단을 보수적으로 제한합니다."
                error = f"{type(exc).__name__}: {exc}"
            results[ticker] = {
                "ticker": ticker, "name": name, "sentiment": sentiment,
                "positive_hits": positive, "negative_hits": negative,
                "summary": summary, "headlines": headlines, "error": error,
            }
        return results


class LLMRebalanceAdvisor:
    def __init__(self, *, model: str = "gpt-5.6", llm=None):
        self.llm = llm or ChatOpenAI(
            model=model, temperature=0, use_responses_api=True
        )

    def propose(self, snapshot: dict) -> RebalanceProposal:
        structured = self.llm.with_structured_output(RebalanceProposal)
        prompt = """
            1. ROLE
                당신은 한국 주식시장을 운용하는 전문 Portfolio Manager이자 주식 투자 의사결정 Agent이다.
                당신의 목표는 단순히 상승 가능성이 높은 종목을 찾는 것이 아니라, 주어진 Universe와 현재 Portfolio를 비교하여 위험 대비 시장 초과수익(Alpha)을 극대화하도록 Portfolio를 리밸런싱하는 것이다.
                매 리밸런싱 시점마다 기존 보유종목에 대한 보유 편향(Endowment Bias)을 제거하고, 모든 종목을 동일한 기준으로 평가한다.
                그러나 작은 순위 변화만으로 기존 종목을 빈번하게 교체해서도 안 된다. 거래비용, Slippage, Turnover 및 단기 시장 Noise를 고려하여 새로운 종목으로 교체할 충분한 근거가 있을 때만 교체한다. "
            2. Primary Objective
                Portfolio의 목적은 다음 우선순위를 따른다.
                향후 투자기간의 시장 대비 초과수익률(Alpha) 극대화
                큰 손실 및 Drawdown 통제
                높은 확률의 투자기회에 자본 집중
                불필요한 Portfolio Turnover 최소화
                특정 종목 및 Sector에 대한 과도한 Concentration 방지
            4. REBALANCING PHILOSOPHY
                리밸런싱 시 반드시 다음 질문을 순서대로 검토한다.
                Question 1 — Market
                현재 시장은 Risk-On, Neutral, Risk-Off 중 어디에 가까운가?
                시장 환경에 따라 신규매수 강도와 Portfolio 위험 수준을 조절한다.
                Question 2 — Existing Holdings
                각 기존 보유종목에 대해 다음을 판단한다.
                "오늘 이 종목을 보유하지 않고 있었다면 현재 정보만 가지고 신규 매수할 것인가?"
                단, 이 질문만으로 즉시 매도하지 않는다.
                기존 종목의 투자 Thesis가 유지되고 있고 상대순위가 충분히 높다면 HOLD를 우선한다.
                Question 3 — Replacement
                신규 후보종목이 기존 보유종목보다 단순히 점수가 높은 것이 아니라,
                거래비용과 불확실성을 감수하면서 교체할 만큼 충분히 우월한가?
                를 판단한다.
            5. HYSTERESIS RULE
                Portfolio Turnover를 억제하기 위해 신규 진입 기준과 기존 보유 유지 기준을 다르게 적용한다.
                기본 원칙:
                Universe 상위권 → 신규매수 후보
                기존 보유종목은 신규매수 기준보다 다소 낮은 순위까지 HOLD 허용
                순위가 크게 하락한 기존 종목 → SELL/REPLACE 후보
                투자 Thesis 또는 Risk 조건이 명백하게 훼손된 경우 → 순위와 관계없이 SELL 검토
                예를 들어 Portfolio가 5종목이라면 다음과 같은 개념을 사용할 수 있다.
                Rank 1~5: Strong BUY Candidate
                Rank 6~15: 기존 보유라면 HOLD 가능
                Rank 16~25: REVIEW / REPLACE Candidate
                Rank >25: SELL Candidate
                단, 위 숫자를 절대적인 규칙으로 사용하지 말고 Universe 규모와 입력 데이터에 맞게 판단한다.
                차이가 미미하면 기존 종목을 유지한다.
                단순히 Total Score가 높은 종목을 기계적으로 선택하지 않는다.
            6. SELL DECISION
                매도는 다음 세 가지로 구분한다.
                HARD SELL
                    투자 Thesis가 명백히 훼손되었거나 심각한 Risk가 발생한 경우.
                    예:
                        중대한 Fundamental deterioration
                        강한 하락추세 전환
                        예상하지 못한 심각한 악재
                        대규모 외국인/기관 이탈
                        Risk management 기준 위반
                REPLACE
                    종목 자체가 나쁘지는 않지만 Universe 내 훨씬 좋은 투자기회가 존재하는 경우.
                    다음 질문을 사용한다.
                    "현재 보유종목을 계속 보유하는 것보다 신규 후보로 교체했을 때 기대되는 추가 Alpha가 충분한가?"

                HOLD
                    신규 후보보다 점수가 약간 낮더라도 다음 조건이라면 유지할 수 있다.
                        투자 Thesis 유지
                        추세 유지
                        수급 악화 없음
                        Score 차이가 크지 않음
                        교체에 따른 기대수익 개선이 작음
            7. BUY DECISION

                신규매수 후보는 단순 Total Score 순위만으로 결정하지 않는다.

                다음을 종합적으로 판단한다.

                Total Score
                Fundamental Quality
                Technical Trend
                Foreign/Institutional Flow
                News/Sentiment
                Score Momentum
                Sector Strength
                Risk/Reward
                기존 Portfolio와의 Diversification
                기존 보유종목 대비 상대적 매력도

                특히 **"무엇을 살 것인가?"보다 "어떤 기존 종목을 대신해서 살 것인가?"**를 명확하게 판단한다.
            11. ANTI-OVERTRADING RULE

                리밸런싱은 거래를 발생시키기 위한 작업이 아니다.
                기존 Portfolio가 여전히 경쟁력이 있다면:
                NO CHANGE
                결정을 적극적으로 허용한다.
                신규 후보의 Score가 기존 종목보다 약간 높다는 이유만으로 교체하지 않는다.
                교체에는 명확한 기대효과가 있어야 한다.

            12. UNCERTAINTY
                데이터가 부족하거나 서로 충돌하는 경우 확신도를 낮춘다.

                예:

                Fundamental = 매우 긍정적
                Technical = 부정적
                Flow = 부정적
                News = 긍정적

                이라면 이를 단순 평균으로 상쇄시키지 말고 의견 충돌 자체를 Risk로 인식한다.

                근거가 부족하면 억지로 결론을 만들지 말고:
                
                HIGH
                MEDIUM
                LOW
                
                Confidence를 명시한다.
            \nDATA:\n
        """  + json.dumps(snapshot, ensure_ascii=False, default=str)
        result = structured.invoke(prompt)
        return result if isinstance(result, RebalanceProposal) else RebalanceProposal.model_validate(result)

    def revise(
        self,
        snapshot: dict,
        current_proposal: RebalanceProposal | dict,
        user_feedback: str,
    ) -> RebalanceProposal:
        """Revise an existing proposal while retaining structured output."""
        feedback = user_feedback.strip()
        if not feedback:
            raise ValueError("리밸런싱 수정 의견이 비어 있습니다.")
        current = (
            current_proposal.model_dump()
            if isinstance(current_proposal, RebalanceProposal)
            else current_proposal
        )
        structured = self.llm.with_structured_output(RebalanceProposal)
        prompt = (
            "당신은 한국 주식 포트폴리오 매니저다. 기존 리밸런싱 제안을 "
            "사용자의 검토 의견에 맞게 수정하라. 입력 snapshot 밖의 종목을 새로 "
            "추가하지 말고, 사용자의 의견과 충돌하지 않는 기존 판단은 유지하라. "
            "수정된 전체 제안을 RebalanceProposal 스키마로 반환하라.\n\n"
            f"USER_FEEDBACK:\n{feedback}\n\n"
            f"CURRENT_PROPOSAL:\n{json.dumps(current, ensure_ascii=False, default=str)}\n\n"
            f"SNAPSHOT:\n{json.dumps(snapshot, ensure_ascii=False, default=str)}"
        )
        result = structured.invoke(prompt)
        return (
            result if isinstance(result, RebalanceProposal)
            else RebalanceProposal.model_validate(result)
        )


class RebalanceValidator:
    def __init__(self, config):
        self.config = config

    def validate(self, snapshot: dict, proposal: RebalanceProposal) -> dict:
        equity = float(snapshot["portfolio"]["total_equity"])
        if equity <= 0:
            error = "총평가금액이 올바르지 않습니다."
            return {
                "approved": False, "override_allowed": False,
                "errors": [error], "hard_errors": [error], "orders": [],
            }
        positions = {item["ticker"]: item for item in snapshot["positions"]}
        top10 = {item["ticker"]: item for item in snapshot["top10"]}
        errors: list[str] = []
        hard_errors: list[str] = []
        orders: list[dict] = []
        turnover = 0.0
        seen: set[str] = set()
        for action in proposal.actions:
            ticker = action.ticker.upper()
            if ticker in seen:
                error = f"{ticker}: 중복 제안"
                errors.append(error)
                hard_errors.append(error)
                continue
            seen.add(ticker)
            held = positions.get(ticker)
            candidate = top10.get(ticker)
            if action.confidence < self.config.rebalance_min_confidence:
                continue
            if action.action == "BUY" and not held and candidate is None:
                error = f"{ticker}: Top10 외 신규 매수"
                errors.append(error)
                hard_errors.append(error)
                continue
            if action.target_weight_pct > self.config.rebalance_max_position_pct * 100:
                error = f"{ticker}: 종목 최대 비중 초과"
                errors.append(error)
                hard_errors.append(error)
                continue
            price = float((held or candidate or {}).get("current_price") or 0)
            if price <= 0:
                error = f"{ticker}: 현재가 없음"
                errors.append(error)
                hard_errors.append(error)
                continue
            current_qty = int((held or {}).get("quantity", 0))
            target_qty = int((equity * action.target_weight_pct / 100) // price)
            delta = target_qty - current_qty
            if action.action in {"SELL", "REDUCE"}:
                delta = -current_qty if action.action == "SELL" else min(0, delta)
            elif action.action == "BUY":
                delta = max(0, delta)
            else:
                delta = 0
            if delta == 0:
                continue
            value = abs(delta) * price
            turnover += value
            orders.append({
                "ticker": ticker,
                "name": (held or candidate or {}).get("name") or stock_name(ticker),
                "side": "BUY" if delta > 0 else "SELL",
                "quantity": abs(delta),
                "original_quantity": current_qty,
                "target_quantity": target_qty,
                "price": price,
                "estimated_value": value,
                "current_weight_pct": float((held or {}).get("weight_pct", 0)),
                "target_weight_pct": action.target_weight_pct,
                "confidence": action.confidence,
                "reason": action.reason,
                "sector": (held or candidate or {}).get("sector", "UNKNOWN"),
            })
        turnover_pct = turnover / equity
        if turnover_pct > self.config.rebalance_max_turnover_pct:
            errors.append(
                f"예상 회전율 {turnover_pct:.1%}가 한도 "
                f"{self.config.rebalance_max_turnover_pct:.1%}를 초과합니다."
            )
        sells = sum(item["estimated_value"] for item in orders if item["side"] == "SELL")
        buys = sum(item["estimated_value"] for item in orders if item["side"] == "BUY")
        sector_values: dict[str, float] = {}
        for item in positions.values():
            sector = str(item.get("sector") or "UNKNOWN")
            sector_values[sector] = sector_values.get(sector, 0.0) + float(
                item.get("market_value")
                or float(item.get("current_price", 0)) * int(item.get("quantity", 0))
            )
        for item in orders:
            sector = str(item.get("sector") or "UNKNOWN")
            change = item["estimated_value"] * (1 if item["side"] == "BUY" else -1)
            sector_values[sector] = max(0.0, sector_values.get(sector, 0.0) + change)
        for sector, value in sector_values.items():
            if sector != "UNKNOWN" and value / equity > self.config.rebalance_max_sector_pct:
                error = f"{sector}: 섹터 최대 비중 초과"
                errors.append(error)
                hard_errors.append(error)
        projected_cash = float(snapshot["portfolio"]["cash"]) + sells - buys
        required_cash_pct = max(
            self.config.rebalance_min_cash_pct,
            proposal.recommended_cash_pct / 100,
        )
        if projected_cash / equity < required_cash_pct:
            errors.append(
                f"예상 현금 비중이 요구 비중 {required_cash_pct:.1%}보다 낮습니다."
            )
        return {
            "approved": not errors,
            "override_allowed": bool(errors) and not hard_errors and bool(orders),
            "errors": errors,
            "hard_errors": hard_errors,
            "orders": sorted(orders, key=lambda item: item["side"], reverse=True),
            "turnover_pct": turnover_pct,
            "projected_cash": projected_cash,
            "projected_cash_pct": projected_cash / equity,
            "required_cash_pct": required_cash_pct,
        }

    def validate_reviewed_orders(
        self,
        snapshot: dict,
        proposal: RebalanceProposal | dict,
        reviewed_orders: list[dict],
    ) -> dict:
        """Revalidate user-approved/edited orders against current policy limits."""
        equity = float(snapshot["portfolio"]["total_equity"])
        positions = {item["ticker"]: item for item in snapshot["positions"]}
        top10 = {item["ticker"]: item for item in snapshot["top10"]}
        errors: list[str] = []
        hard_errors: list[str] = []
        orders: list[dict] = []
        seen: set[str] = set()
        for raw in reviewed_orders:
            ticker = str(raw.get("ticker") or "").upper()
            side = str(raw.get("side") or "").upper()
            try:
                quantity = int(raw.get("quantity", 0))
            except (TypeError, ValueError):
                quantity = 0
            held, candidate = positions.get(ticker), top10.get(ticker)
            if not ticker or ticker in seen:
                error = f"{ticker or '-'}: 중복 또는 빈 종목코드"
                errors.append(error); hard_errors.append(error)
                continue
            seen.add(ticker)
            if side not in {"BUY", "SELL"} or quantity <= 0:
                error = f"{ticker}: 주문구분 또는 수량 오류"
                errors.append(error); hard_errors.append(error)
                continue
            if side == "BUY" and held is None and candidate is None:
                error = f"{ticker}: Top10 외 신규 매수"
                errors.append(error); hard_errors.append(error)
                continue
            current_qty = int((held or {}).get("quantity", 0))
            if side == "SELL" and (held is None or quantity > current_qty):
                error = f"{ticker}: 보유수량을 초과한 매도"
                errors.append(error); hard_errors.append(error)
                continue
            source = held or candidate or {}
            price = float(source.get("current_price") or raw.get("price") or 0)
            if price <= 0:
                error = f"{ticker}: 현재가 없음"
                errors.append(error); hard_errors.append(error)
                continue
            target_qty = current_qty + quantity if side == "BUY" else current_qty - quantity
            target_weight_pct = target_qty * price / equity * 100 if equity else 0
            if target_weight_pct > self.config.rebalance_max_position_pct * 100:
                error = f"{ticker}: 종목 최대 비중 초과"
                errors.append(error); hard_errors.append(error)
                continue
            orders.append({
                **raw,
                "ticker": ticker,
                "name": source.get("name") or raw.get("name") or stock_name(ticker),
                "side": side,
                "quantity": quantity,
                "original_quantity": current_qty,
                "target_quantity": target_qty,
                "price": price,
                "estimated_value": quantity * price,
                "current_weight_pct": float((held or {}).get("weight_pct", 0)),
                "target_weight_pct": target_weight_pct,
                "sector": source.get("sector", raw.get("sector", "UNKNOWN")),
            })

        turnover = sum(item["estimated_value"] for item in orders)
        turnover_pct = turnover / equity if equity else 0
        if turnover_pct > self.config.rebalance_max_turnover_pct:
            errors.append(
                f"예상 회전율 {turnover_pct:.1%}가 한도 "
                f"{self.config.rebalance_max_turnover_pct:.1%}를 초과합니다."
            )
        sells = sum(item["estimated_value"] for item in orders if item["side"] == "SELL")
        buys = sum(item["estimated_value"] for item in orders if item["side"] == "BUY")
        sector_values: dict[str, float] = {}
        for item in positions.values():
            sector = str(item.get("sector") or "UNKNOWN")
            sector_values[sector] = sector_values.get(sector, 0) + float(item["market_value"])
        for item in orders:
            sector = str(item.get("sector") or "UNKNOWN")
            direction = 1 if item["side"] == "BUY" else -1
            sector_values[sector] = max(
                0, sector_values.get(sector, 0) + direction * item["estimated_value"]
            )
        for sector, value in sector_values.items():
            if sector != "UNKNOWN" and equity and value / equity > self.config.rebalance_max_sector_pct:
                error = f"{sector}: 섹터 최대 비중 초과"
                errors.append(error); hard_errors.append(error)
        recommended_cash_pct = float(
            proposal.recommended_cash_pct
            if isinstance(proposal, RebalanceProposal)
            else proposal.get("recommended_cash_pct", 0)
        )
        required_cash_pct = max(
            self.config.rebalance_min_cash_pct, recommended_cash_pct / 100
        )
        projected_cash = float(snapshot["portfolio"]["cash"]) + sells - buys
        if equity and projected_cash / equity < required_cash_pct:
            errors.append(
                f"예상 현금 비중이 요구 비중 {required_cash_pct:.1%}보다 낮습니다."
            )
        return {
            "approved": not errors,
            "override_allowed": bool(errors) and not hard_errors and bool(orders),
            "errors": errors,
            "hard_errors": hard_errors,
            "orders": sorted(orders, key=lambda item: item["side"], reverse=True),
            "turnover_pct": turnover_pct,
            "projected_cash": projected_cash,
            "projected_cash_pct": projected_cash / equity if equity else 0,
            "required_cash_pct": required_cash_pct,
            "individually_reviewed": True,
        }


def proposal_id(payload: dict) -> str:
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode()
    ).hexdigest()[:8].upper()
    return f"RB-{datetime.now().strftime('%Y%m%d')}-{digest}"


class RebalanceExecutor:
    def __init__(self, controller):
        self.controller = controller

    def execute(self, package: dict) -> dict:
        results: list[dict] = []
        orders = package["validation"]["orders"]
        sell_tickers = {
            row["ticker"] for row in orders if row["side"] == "SELL"
        }
        # Never submit another sell while a previous rebalance/order is still
        # reconcilable for the same ticker. Unrelated historical orders must
        # not block a BUY-only rebalance.
        blocking_orders = (
            self._wait_for_pending_orders(sell_tickers) if sell_tickers else []
        )
        if blocking_orders:
            return {
                "status": "AWAITING_SELL_FILLS", "orders": results,
                "blocking_orders": self._blocking_order_rows(blocking_orders),
            }
        for item in (row for row in orders if row["side"] == "SELL"):
            position = self.controller.context.broker.get_position(item["ticker"])
            current_quantity = position.quantity if position is not None else 0
            target_quantity = self._target_quantity(item, package)
            quantity = max(0, current_quantity - target_quantity)
            if quantity <= 0:
                results.append({**item, "quantity": 0, "status": "ALREADY_APPLIED"})
                continue
            result = self.controller.manual_sell(
                item["ticker"], quantity, limit_price=self.controller.quote(item["ticker"])
            )
            results.append({
                **item, "quantity": quantity, "price": result.price,
                "status": result.status, "order_id": result.order_id,
            })

        blocking_orders = (
            self._wait_for_pending_orders(sell_tickers) if sell_tickers else []
        )
        if blocking_orders:
            return {
                "status": "AWAITING_SELL_FILLS", "orders": results,
                "blocking_orders": self._blocking_order_rows(blocking_orders),
            }

        for item in (row for row in orders if row["side"] == "BUY"):
            balance = self.controller.context.broker.get_balance()
            price = self.controller.quote(item["ticker"])
            affordable = int(float(balance.get("cash", 0)) // price)
            position = self.controller.context.broker.get_position(item["ticker"])
            current_quantity = position.quantity if position is not None else 0
            target_quantity = self._target_quantity(item, package)
            quantity = min(max(0, target_quantity - current_quantity), affordable)
            if quantity <= 0:
                status = "ALREADY_APPLIED" if current_quantity >= target_quantity else "SKIPPED_NO_CASH"
                results.append({**item, "quantity": 0, "status": status})
                continue
            result = self.controller.manual_buy(
                item["ticker"], quantity, limit_price=price,
                sector=item.get("sector", "UNKNOWN"),
                trailing_stop_pct=self.controller.context.config.trailing_stop_pct,
            )
            results.append({**item, "quantity": quantity, "price": price,
                            "status": result.status, "order_id": result.order_id})
        return {"status": "ORDERS_SUBMITTED", "orders": results}

    def _wait_for_pending_orders(self, sell_tickers: set[str]) -> list[dict]:
        def relevant_orders() -> list[dict]:
            return [
                order
                for order in self.controller.context.store.list_reconcilable_orders()
                if order.get("side") == "SELL"
                and order.get("ticker") in sell_tickers
            ]

        if not relevant_orders():
            return []
        if self.controller.service.reconciler is None:
            return relevant_orders()
        deadline = time.monotonic() + self.controller.context.config.rebalance_fill_wait_seconds
        while relevant_orders() and time.monotonic() < deadline:
            if self.controller.service.reconciler is not None:
                self.controller.service.reconciler.reconcile()
            time.sleep(1)
        return relevant_orders()

    @staticmethod
    def _blocking_order_rows(orders: list[dict]) -> list[dict]:
        return [{
            "ticker": order.get("ticker", ""),
            "side": order.get("side", ""),
            "status": order.get("status", ""),
            "quantity": int(order.get("payload", {}).get("quantity", 0) or 0),
            "broker_order_id": order.get("broker_order_id") or "",
            "updated_at": order.get("updated_at", ""),
        } for order in orders]

    @staticmethod
    def _target_quantity(item: dict, package: dict) -> int:
        if "target_quantity" in item:
            return int(item["target_quantity"])
        original = next(
            (
                int(position.get("quantity", 0))
                for position in package.get("snapshot", {}).get("positions", [])
                if position.get("ticker") == item.get("ticker")
            ),
            0,
        )
        return (
            max(0, original - int(item["quantity"]))
            if item["side"] == "SELL"
            else original + int(item["quantity"])
        )
