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


def _ml_probability(value, default: float = 0.0) -> float:
    """Keep model scores in [0, 1] without factor-score rounding loss."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return round(max(0.0, min(1.0, number)), 6) if math.isfinite(number) else default


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
        limit: int | None = None,
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
        final_limit = limit or self.context.config.recommendation_final_limit
        analysis_shortlist = self.context.config.recommendation_analysis_shortlist
        version = self.context.config.strategy_version
        if self.context.config.market_region == "US":
            version = f"{version}:US"

        # KR menu 10 always uses the trained ML model.  TradingController also
        # exposes KR ML Filter as always enabled, so do not let a legacy false
        # value in system_controls silently switch recommendations to the
        # market-cap fallback (whose ML scores are necessarily all zero).
        ml_enabled = self.context.config.market_region == "KR"
        candidate_source = "ML_SNAPSHOT" if ml_enabled else "MARKET_CAP_UNIVERSE"
        cache_version = (
            f"{version}:recommendation:{candidate_source.lower()}:"
            f"n{self.context.config.recommendation_universe_per_market}:"
            f"analysis-{analysis_shortlist}:final-{final_limit}:"
            "detail-v2:ml-score-v2:"
            f"scope-{scope.lower()}"
        )
        if not refresh:
            cached = self.context.store.get_recommendations(date_text, cache_version)
            if cached is not None:
                return cached[:final_limit]

        if ml_enabled:
            # Menu 10 needs a broad universe here.  The pre-open session only
            # stores the small order-entry candidate set, so reusing it would
            # cap recommendation analysis before the four-factor shortlist.
            candidates = self._ml_snapshot_candidates(
                trade_date,
                limit=None,
                per_market=self.context.config.recommendation_universe_per_market,
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
        # Stage 1: technical/fundamental/news/flow analysis chooses 30 stocks.
        factor_shortlist = sorted(
            results, key=lambda item: item["total_score"], reverse=True
        )[:analysis_shortlist]

        # Stage 2: only the 30-stock shortlist is passed through the ML filter.
        # When the operator disables ML Filter, preserve the factor ordering.
        if ml_enabled:
            factor_shortlist.sort(
                key=lambda item: (
                    float(item.get("ml_score", 0.0)),
                    float(item.get("classification_probability", 0.0)),
                ),
                reverse=True,
            )
        selected = factor_shortlist[:final_limit]
        for rank, item in enumerate(selected, 1):
            item["rank"] = rank
        self.context.store.save_recommendations(date_text, cache_version, selected)
        self.context.store.audit(
            "TOP_RECOMMENDATIONS_CREATED",
            date_text,
            {
                "candidate_count": total,
                "analysis_shortlist_count": len(factor_shortlist),
                "selected_count": len(selected),
                "candidate_source": candidate_source,
                "ml_filter_enabled": ml_enabled,
                "universe_scope": scope,
            },
        )
        return selected

    def _ml_snapshot_candidates(
        self,
        trade_date: date,
        *,
        limit: int | None = None,
        per_market: int | None = None,
    ) -> list[dict]:
        candidates = self.context.candidate_provider.candidates(
            trade_date,
            per_market or self.context.config.recommendation_universe_per_market,
        )
        missing_scores = [
            str(candidate.get("ticker", "UNKNOWN"))
            for candidate in candidates
            if candidate.get("ml_score") is None
        ]
        if missing_scores:
            preview = ", ".join(missing_scores[:5])
            raise ValueError(
                "ML prediction score is missing for recommendation candidates: "
                f"{preview}"
            )
        for candidate in candidates:
            candidate.setdefault(
                "classification_probability", candidate["ml_score"]
            )
            candidate.setdefault("ml_rank", 9999)
        if limit is not None:
            candidates = sorted(
                candidates,
                key=lambda item: (
                    float(item.get("ml_score", 0.0)),
                    float(item.get("classification_probability", 0.0)),
                ),
                reverse=True,
            )[:limit]
        for candidate in candidates:
            candidate.setdefault("candidate_source", "ML_SNAPSHOT")
            candidate.setdefault("model_version", 0)
        return candidates

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
            self._technical_reason,
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
            "ml_score": _ml_probability(candidate.get("ml_score", 0)),
            "classification_probability": _ml_probability(
                candidate.get("classification_probability", 0)
            ),
            "ml_rank": int(candidate.get("ml_rank", 9999)),
            "technical_reason": technical_reason,
            "fundamental_reason": fundamental_reason,
            "news_reason": news_reason,
            "flow_reason": flow_reason,
            "recommendation_reason": " · ".join(
                f"{label} {score:.1f}" for label, score in strongest
            ),
        }

    @staticmethod
    def _technical_reason(value: dict) -> str:
        """Format the technical indicators used by the recommendation score."""
        indicators = value.get("indicators") or {}

        def number(*keys: str, digits: int = 2) -> str:
            for key in keys:
                raw = indicators.get(key)
                if raw is not None:
                    try:
                        return f"{float(raw):.{digits}f}"
                    except (TypeError, ValueError):
                        return str(raw)
            return "-"

        return " · ".join([
            str(value.get("signal", "NEUTRAL")),
            f"RSI {number('RSI', digits=1)}",
            f"ADX {number('ADX', digits=1)}",
            f"DI+ {number('DI_PLUS', 'DI+', digits=1)}",
            f"DI- {number('DI_MINUS', 'DI-', digits=1)}",
            #f"MACD {number('MACD', digits=3)}",
            f"MACD Signal {number('MACD_SIGNAL', digits=3)}",
            f"Volume Ratio {number('VOLUME_RATIO', 'volume_ratio', digits=2)}",
            # f"MA5/20/60 {number('MA5', digits=0)}/{number('MA20', digits=0)}/{number('MA60', digits=0)}",
            f"ATR {number('ATR', digits=2)}",
        ])

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
