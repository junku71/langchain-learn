from __future__ import annotations

import math
from datetime import date
from pathlib import Path
from typing import Callable

import pandas as pd

from analysis.flow import analyze_flow
from analysis.fundamental import analyze_fundamental
from analysis.news_service import NewsAnalysisService
from analysis.technical import get_technical_analysis
from analysis.ticker_mapper import TickerMapper
from analysis.us_market import analyze_us_fundamental, analyze_us_participation
from trading.display import stock_name
from trading.graphs import TradingGraphContext
from trading.market import get_market_profile


POSITIVE_WORDS = (
    "호실적", "상승", "성장", "수주", "흑자", "상향", "최고", "개선",
    "확대", "돌파", "강세", "기대",
)
NEGATIVE_WORDS = (
    "하락", "감소", "적자", "소송", "조사", "리콜", "하향", "부진",
    "우려", "약세", "쇼크", "중단",
)


def _score(value, default: float = 50.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return round(max(0.0, min(100.0, number)), 1) if math.isfinite(number) else default


class RecommendationService:
    WEIGHTS = {
        "technical_score": 0.30,
        "fundamental_score": 0.25,
        "news_score": 0.20,
        "flow_score": 0.25,
    }

    def __init__(
        self,
        context: TradingGraphContext,
        *,
        technical_fn: Callable = get_technical_analysis,
        fundamental_fn: Callable | None = None,
        flow_fn: Callable | None = None,
        news_service: NewsAnalysisService | None = None,
        universe_path: str | Path = "data/ml/universe_top200.csv",
    ):
        self.context = context
        self.technical_fn = technical_fn
        us_market = context.config.market_region == "US"
        self.fundamental_fn = fundamental_fn or (
            analyze_us_fundamental if us_market else
            lambda ticker: analyze_fundamental(ticker, provider=context.broker)
        )
        self.flow_fn = flow_fn or (
            analyze_us_participation if us_market else
            lambda ticker: analyze_flow(ticker, lookback=5, provider=context.broker)
        )
        self.news_service = news_service or NewsAnalysisService()
        self.universe_path = Path(universe_path)

    def top_recommendations(
        self,
        trade_date: date,
        *,
        limit: int = 10,
        universe_scope: str = "BOTH",
        refresh: bool = False,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> list[dict]:
        scope = universe_scope.strip().upper()
        profile = get_market_profile(self.context.config.market_region)
        allowed = {*profile.universes, "BOTH"}
        if scope not in allowed:
            raise ValueError(f"universe_scope must be one of {sorted(allowed)}")
        date_text = trade_date.isoformat()
        version = self.context.config.strategy_version
        if self.context.config.market_region == "US":
            version = f"{version}:US"
        ml_enabled = self.context.config.market_region == "KR" and self.context.store.get_bool_control(
            "ml_filter_enabled", self.context.config.ml_filter_enabled
        )
        candidate_source = "ML_SNAPSHOT" if ml_enabled else "MARKET_CAP_UNIVERSE"
        cache_version = (
            f"{version}:recommendation:{candidate_source.lower()}:"
            f"n{self.context.config.recommendation_universe_per_market}:"
            f"scope-{scope.lower()}"
        )
        if not refresh:
            cached = self.context.store.get_recommendations(date_text, cache_version)
            if cached is not None:
                return cached[:limit]

        if ml_enabled:
            session_id = f"{date_text}:{version}"
            session = self.context.store.get_session(session_id)
            candidates = list(
                ((session or {}).get("payload") or {}).get("candidates", [])
            )
            if not candidates:
                candidates = self.context.candidate_provider.candidates(
                    trade_date, self.context.config.max_candidates_per_market
                )
        else:
            candidates = self._market_cap_candidates(scope)
        if scope != "BOTH":
            candidates = [
                candidate for candidate in candidates
                if str(candidate.get("market") or "").upper() == scope
            ]
        if not candidates:
            return []
        for candidate in candidates:
            candidate["candidate_source"] = candidate_source

        results = []
        total = len(candidates)
        for index, candidate in enumerate(candidates, 1):
            ticker = str(candidate["ticker"])
            if progress:
                progress(index, total, ticker)
            results.append(self._analyze(candidate))
        results.sort(
            key=lambda item: (item["total_score"], item.get("ml_score", 0)),
            reverse=True,
        )
        for rank, item in enumerate(results, 1):
            item["rank"] = rank
        selected = results[:limit]
        self.context.store.save_recommendations(date_text, cache_version, selected)
        self.context.store.audit(
            "TOP_RECOMMENDATIONS_CREATED",
            date_text,
            {
                "candidate_count": total,
                "selected_count": len(selected),
                "candidate_source": candidate_source,
                "universe_scope": scope,
            },
        )
        return selected

    def _market_cap_candidates(self, universe_scope: str = "BOTH") -> list[dict]:
        if self.context.config.market_region == "US":
            return self._us_candidates(universe_scope)
        if not self.universe_path.exists():
            raise ValueError(
                f"Non-ML universe file not found: {self.universe_path}. "
                "Run the universe data collection first."
            )
        universe = pd.read_csv(self.universe_path, dtype={"ticker": str})
        required = {"market", "ticker", "name", "market_cap_rank"}
        missing = required - set(universe.columns)
        if missing:
            raise ValueError(f"Universe file missing columns: {sorted(missing)}")
        universe["market_cap_rank"] = pd.to_numeric(
            universe["market_cap_rank"], errors="coerce"
        )
        universe = universe.dropna(subset=["ticker", "market_cap_rank"])
        scope = universe_scope.strip().upper()
        if scope != "BOTH":
            universe = universe[
                universe["market"].astype(str).str.upper() == scope
            ]
        universe = universe.sort_values(["market", "market_cap_rank"])
        universe = universe.groupby("market", group_keys=False).head(
            self.context.config.recommendation_universe_per_market
        )
        universe["sector"] = universe.get(
            "sector", pd.Series(index=universe.index, dtype=object)
        ).fillna("UNKNOWN")
        universe["ml_score"] = 0.0
        return universe[
            ["ticker", "name", "market", "sector", "market_cap_rank", "ml_score"]
        ].to_dict("records")

    def _us_candidates(self, universe_scope: str) -> list[dict]:
        markets = (
            get_market_profile("US").universes
            if universe_scope == "BOTH" else (universe_scope,)
        )
        rows: list[dict] = []
        mapper = TickerMapper()
        limit = self.context.config.recommendation_universe_per_market
        for market in markets:
            frame = mapper.load_market(market).head(limit)
            for rank, item in enumerate(frame.to_dict("records"), 1):
                rows.append({
                    "ticker": item["ticker"], "name": item["name"],
                    "market": market, "sector": "UNKNOWN",
                    "market_cap_rank": rank, "ml_score": 0.0,
                })
        return rows

    def _factor(self, callback: Callable[[], dict], formatter: Callable[[dict], str]):
        try:
            result = callback()
            return _score(result.get("score", result.get("technical_score", 50))), formatter(result)
        except Exception as error:
            return 50.0, f"데이터 오류 {type(error).__name__}"

    def _analyze(self, candidate: dict) -> dict:
        ticker = str(candidate["ticker"])
        technical_score, technical_reason = self._factor(
            lambda: self.technical_fn(ticker),
            lambda value: (
                f"{value.get('signal', 'NEUTRAL')} RSI "
                f"{float((value.get('indicators') or {}).get('RSI', 0)):.1f}"
            ),
        )
        fundamental_score, fundamental_reason = self._factor(
            lambda: self.fundamental_fn(ticker),
            lambda value: (
                f"{value.get('signal', 'NEUTRAL')} PER {value.get('PER') or '-'} "
                f"PBR {value.get('PBR') or '-'} ROE {value.get('ROE') or '-'}"
            ),
        )
        flow_score, flow_reason = self._factor(
            lambda: self.flow_fn(ticker),
            lambda value: (
                f"{value.get('signal', 'NEUTRAL')} 동반매수 "
                f"{value.get('joint_buy_days', 0)}일 외국인 "
                f"{float(value.get('foreign_net_sum', 0)):,.0f} 기관 "
                f"{float(value.get('institution_net_sum', 0)):,.0f}"
            ),
        )
        try:
            news = self.news_service.collect(ticker)
            news_score, news_reason = self._news_score(news)
        except Exception as error:
            news_score, news_reason = 50.0, f"데이터 오류 {type(error).__name__}"

        scores = {
            "technical_score": technical_score,
            "fundamental_score": fundamental_score,
            "news_score": news_score,
            "flow_score": flow_score,
        }
        total = round(sum(scores[key] * weight for key, weight in self.WEIGHTS.items()), 1)
        strongest = sorted(
            (("기술", technical_score), ("기본", fundamental_score),
             ("뉴스", news_score), ("수급", flow_score)),
            key=lambda item: item[1], reverse=True,
        )[:2]
        return {
            "rank": 0,
            "ticker": ticker,
            "name": candidate.get("name") or stock_name(ticker),
            "market": candidate.get("market", ""),
            "sector": candidate.get("sector", "UNKNOWN"),
            "candidate_source": candidate.get("candidate_source", "UNKNOWN"),
            "total_score": total,
            **scores,
            "ml_score": _score(candidate.get("ml_score", 0), 0),
            "technical_reason": technical_reason,
            "fundamental_reason": fundamental_reason,
            "news_reason": news_reason,
            "flow_reason": flow_reason,
            "recommendation_reason": " · ".join(
                f"{label} {score:.1f}" for label, score in strongest
            ),
        }

    @staticmethod
    def _news_score(data: dict) -> tuple[float, str]:
        items = [
            *data.get("naver_news", []),
            *data.get("naver_earnings_news", []),
            *data.get("yahoo_news", []),
        ]
        text = " ".join(
            f"{item.get('title', '')} {item.get('description', '')}" for item in items
        )
        positive = sum(text.count(word) for word in POSITIVE_WORDS)
        negative = sum(text.count(word) for word in NEGATIVE_WORDS)
        score = 50.0 + max(-20, min(20, (positive - negative) * 3))
        features = data.get("earnings_features") or {}
        revision = features.get("eps_revision_30d")
        upside = features.get("target_upside")
        balance = features.get("revision_balance")
        if revision is not None:
            score += max(-12, min(12, float(revision) * 100))
        if upside is not None:
            score += max(-12, min(12, float(upside) * 50))
        if balance is not None:
            score += max(-8, min(8, float(balance) * 8))
        score = _score(score)
        return score, (
            f"기사 {len(items)}건 긍정 {positive} 부정 {negative}"
            + (f" 목표상승 {float(upside):.1%}" if upside is not None else "")
        )
