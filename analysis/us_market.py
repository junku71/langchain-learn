from __future__ import annotations

import math

import yfinance as yf


def analyze_us_fundamental(ticker: str) -> dict:
    """Normalize Yahoo company metrics to the existing 0-100 factor contract."""
    info = yf.Ticker(ticker).info
    pe = info.get("trailingPE")
    pb = info.get("priceToBook")
    roe = info.get("returnOnEquity")
    debt = info.get("debtToEquity")
    score = 50.0
    if pe is not None and float(pe) > 0:
        score += 10 if float(pe) <= 25 else (-8 if float(pe) >= 50 else 0)
    if pb is not None and float(pb) > 0:
        score += 6 if float(pb) <= 5 else (-5 if float(pb) >= 12 else 0)
    if roe is not None:
        score += max(-15, min(15, float(roe) * 50))
    if debt is not None:
        score += 5 if float(debt) <= 100 else (-8 if float(debt) >= 250 else 0)
    score = max(0.0, min(100.0, score))
    return {
        "score": round(score, 1),
        "signal": "BUY" if score >= 65 else "SELL" if score <= 35 else "NEUTRAL",
        "PER": pe, "PBR": pb,
        "ROE": None if roe is None else round(float(roe) * 100, 2),
    }


def analyze_us_participation(ticker: str, lookback: int = 20) -> dict:
    """US substitute for KRX investor flow: price/volume participation strength."""
    frame = yf.download(ticker, period="3mo", progress=False, auto_adjust=False)
    if frame.empty or len(frame) < lookback + 1:
        raise ValueError("Insufficient US price/volume history")
    close = frame["Close"].squeeze().astype(float)
    volume = frame["Volume"].squeeze().astype(float)
    momentum = float(close.iloc[-1] / close.iloc[-lookback - 1] - 1)
    recent_volume = float(volume.tail(5).mean())
    baseline_volume = float(volume.tail(lookback).mean())
    ratio = recent_volume / baseline_volume if baseline_volume > 0 else 1.0
    score = 50 + max(-25, min(25, momentum * 100)) + max(-15, min(15, (ratio - 1) * 30))
    if not math.isfinite(score):
        score = 50
    score = max(0.0, min(100.0, score))
    return {
        "score": round(score, 1),
        "signal": "BUY" if score >= 65 else "SELL" if score <= 35 else "NEUTRAL",
        "joint_buy_days": 0,
        "foreign_net_sum": momentum * 100,
        "institution_net_sum": (ratio - 1) * 100,
    }
