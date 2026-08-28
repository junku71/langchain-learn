import pandas as pd
import numpy as np
import yfinance as yf


# -------------------------------------------------
# 1. 주가 데이터 가져오기
# -------------------------------------------------

def get_stock_data(
    ticker: str,
    period: str = "1y",
) -> pd.DataFrame:

    df = yf.download(
        ticker,
        period=period,
        auto_adjust=True,
        progress=False
    )

    # yfinance may return (price, ticker) MultiIndex columns even when only
    # one ticker was requested. Indicators below expect plain OHLCV columns.
    if isinstance(df.columns, pd.MultiIndex):
        ticker_count = df.columns.get_level_values(1).nunique()
        if ticker_count != 1:
            raise ValueError("Expected data for exactly one ticker")
        df.columns = df.columns.get_level_values(0)

    price_columns = ["Open", "High", "Low", "Close"]
    missing_columns = [
        column for column in [*price_columns, "Volume"]
        if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Missing market data columns for {ticker}: "
            f"{', '.join(missing_columns)}"
        )

    # Yahoo can publish a trailing row whose volume is present while every
    # price is NaN. It is not a usable trading session for any calculation.
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=price_columns)

    if df.empty:
        raise ValueError(f"No complete price data available for {ticker}")

    return df


# -------------------------------------------------
# 2. 이동평균
# -------------------------------------------------

def calculate_moving_averages(
    df: pd.DataFrame
) -> pd.DataFrame:

    df = df.copy()

    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA60"] = df["Close"].rolling(60).mean()

    return df


# -------------------------------------------------
# 3. RSI
# -------------------------------------------------

def calculate_rsi(
    df: pd.DataFrame,
    period: int = 14
) -> pd.DataFrame:

    df = df.copy()

    delta = df["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss

    df["RSI"] = 100 - (100 / (1 + rs))

    return df


# -------------------------------------------------
# 4. MACD
# -------------------------------------------------

def calculate_macd(
    df: pd.DataFrame
) -> pd.DataFrame:

    df = df.copy()

    ema12 = df["Close"].ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = df["Close"].ewm(
        span=26,
        adjust=False
    ).mean()

    df["MACD"] = ema12 - ema26

    df["MACD_SIGNAL"] = (
        df["MACD"]
        .ewm(span=9, adjust=False)
        .mean()
    )

    df["MACD_HIST"] = (
        df["MACD"]
        - df["MACD_SIGNAL"]
    )

    return df


# -------------------------------------------------
# 5. ATR
# -------------------------------------------------

def calculate_atr(
    df: pd.DataFrame,
    period: int = 14
) -> pd.DataFrame:

    df = df.copy()

    high_low = (
        df["High"]
        - df["Low"]
    )

    high_close = (
        df["High"]
        - df["Close"].shift()
    ).abs()

    low_close = (
        df["Low"]
        - df["Close"].shift()
    ).abs()

    true_range = pd.concat(
        [
            high_low,
            high_close,
            low_close
        ],
        axis=1
    ).max(axis=1)

    df["ATR"] = (
        true_range
        .rolling(period)
        .mean()
    )

    return df


# -------------------------------------------------
# 6. 거래량 평균
# -------------------------------------------------

def calculate_volume(
    df: pd.DataFrame
) -> pd.DataFrame:

    df = df.copy()

    df["VOLUME_MA20"] = (
        df["Volume"]
        .rolling(20)
        .mean()
    )

    df["VOLUME_RATIO"] = (
        df["Volume"]
        / df["VOLUME_MA20"]
    )

    return df

# -------------------------------------------------
# 7. ADX / DI+ / DI-
# -------------------------------------------------

def calculate_adx(
    df: pd.DataFrame,
    period: int = 14
) -> pd.DataFrame:

    df = df.copy()

    # 이전 고가/저가와의 차이
    high_diff = df["High"].diff()
    low_diff = -df["Low"].diff()

    # +DM
    plus_dm = np.where(
        (high_diff > low_diff) & (high_diff > 0),
        high_diff,
        0.0
    )

    # -DM
    minus_dm = np.where(
        (low_diff > high_diff) & (low_diff > 0),
        low_diff,
        0.0
    )

    plus_dm = pd.Series(
        plus_dm,
        index=df.index
    )

    minus_dm = pd.Series(
        minus_dm,
        index=df.index
    )

    # True Range
    high_low = (
        df["High"]
        - df["Low"]
    )

    high_close = (
        df["High"]
        - df["Close"].shift()
    ).abs()

    low_close = (
        df["Low"]
        - df["Close"].shift()
    ).abs()

    true_range = pd.concat(
        [
            high_low,
            high_close,
            low_close
        ],
        axis=1
    ).max(axis=1)

    # Wilder smoothing
    atr = true_range.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    plus_dm_smoothed = plus_dm.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    minus_dm_smoothed = minus_dm.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    # DI 계산
    df["DI_PLUS"] = (
        100
        * plus_dm_smoothed
        / atr
    )

    df["DI_MINUS"] = (
        100
        * minus_dm_smoothed
        / atr
    )

    # DX 계산
    dx = (
        (
            df["DI_PLUS"]
            - df["DI_MINUS"]
        ).abs()
        /
        (
            df["DI_PLUS"]
            + df["DI_MINUS"]
        )
    ) * 100

    # ADX
    df["ADX"] = dx.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    return df

# -------------------------------------------------
# Pipeline: OHLCV -> Moving Averages -> RSI ->  MACD -> ATR -> Volume
# 나중에 LangGraph에서는 node --> node --> node --> ... 으로 발전함.
# -------------------------------------------------



def calculate_indicators(
    df: pd.DataFrame
) -> pd.DataFrame:

    df = calculate_moving_averages(df)

    df = calculate_rsi(df)

    df = calculate_macd(df)

    df = calculate_atr(df)

    df = calculate_volume(df)

    df = calculate_adx(df)

    return df


# -------------------------------------------------
# 분석 함수
# -------------------------------------------------

def analyze_technical(
    df: pd.DataFrame
) -> dict:

    latest = df.iloc[-1]

    result = {}

    # 이동평균 분석

    if (
        latest["Close"] > latest["MA20"]
        and latest["MA20"] > latest["MA60"]
    ):
        result["trend"] = "bullish"

    elif (
        latest["Close"] < latest["MA20"]
        and latest["MA20"] < latest["MA60"]
    ):
        result["trend"] = "bearish"

    else:
        result["trend"] = "neutral"


    # RSI 분석

    if latest["RSI"] >= 70:
        result["rsi_signal"] = "overbought"

    elif latest["RSI"] <= 30:
        result["rsi_signal"] = "oversold"

    else:
        result["rsi_signal"] = "neutral"


    # MACD 분석

    if latest["MACD"] > latest["MACD_SIGNAL"]:
        result["macd_signal"] = "bullish"

    else:
        result["macd_signal"] = "bearish"


    # 거래량 분석

    if latest["VOLUME_RATIO"] >= 1.5:
        result["volume_signal"] = "high"

    else:
        result["volume_signal"] = "normal"


    # -------------------------
    # ADX 추세 강도
    # -------------------------

    if latest["ADX"] >= 30:
        result["trend_strength"] = "strong"

    elif latest["ADX"] >= 20:
        result["trend_strength"] = "moderate"

    else:
        result["trend_strength"] = "weak"


    # -------------------------
    # DI 방향
    # -------------------------

    if latest["DI_PLUS"] > latest["DI_MINUS"]:
        result["di_signal"] = "bullish"

    elif latest["DI_PLUS"] < latest["DI_MINUS"]:
        result["di_signal"] = "bearish"

    else:
        result["di_signal"] = "neutral"

    return result



# -------------------------------------------------
# 기술적 점수 계산
# -------------------------------------------------

def calculate_technical_score(
    df: pd.DataFrame
) -> int:

    latest = df.iloc[-1]

    score = 50


    # Trend

    if (
        latest["Close"] > latest["MA20"]
        and latest["MA20"] > latest["MA60"]
    ):
        score += 20

    elif (
        latest["Close"] < latest["MA20"]
        and latest["MA20"] < latest["MA60"]
    ):
        score -= 20


    # RSI

    if 50 <= latest["RSI"] <= 70:
        score += 10

    elif latest["RSI"] >= 75:
        score -= 10


    # MACD

    if latest["MACD"] > latest["MACD_SIGNAL"]:
        score += 15

    else:
        score -= 15


    # Volume

    if latest["VOLUME_RATIO"] >= 1.5:
        score += 5

    # -------------------------
    # DI 방향
    # -------------------------

    if latest["DI_PLUS"] > latest["DI_MINUS"]:
        score += 10

    else:
        score -= 10


    # -------------------------
    # ADX 추세 강도
    # -------------------------

    if latest["ADX"] >= 30:

        if latest["DI_PLUS"] > latest["DI_MINUS"]:
            score += 10

        else:
            score -= 10

    elif latest["ADX"] >= 20:

        if latest["DI_PLUS"] > latest["DI_MINUS"]:
            score += 5

        else:
            score -= 5
    # 0~100 제한

    score = max(
        0,
        min(100, score)
    )


    return score

# -------------------------------------------------
# 거래 신호 생성
# -------------------------------------------------

def make_trading_signal(
    score: int
) -> str:

    if score >= 70:

        return "BUY"

    elif score <= 30:

        return "SELL"

    else:

        return "HOLD"

#-------------------------------------------------
# 종합 분석 함수
#-------------------------------------------------

def analyze_stock(
    ticker: str
) -> dict:

    df = get_stock_data(
        ticker
    )

    df = calculate_indicators(
        df
    )

    technical = analyze_technical(
        df
    )

    score = calculate_technical_score(
        df
    )

    signal = make_trading_signal(
        score
    )

    latest = df.iloc[-1]

    result = {

        "ticker": ticker,

        "price": float(
            latest["Close"]
        ),

        "technical_score": score,

        "signal": signal,

        "analysis": technical,

        "indicators": {

            "MA5": float(
                latest["MA5"]
            ),

            "MA20": float(
                latest["MA20"]
            ),

            "MA60": float(
                latest["MA60"]
            ),

            "RSI": float(
                latest["RSI"]
            ),

            "MACD": float(
                latest["MACD"]
            ),

            "ATR": float(
                latest["ATR"]
            ),

            "ADX": float(
                latest["ADX"]
            ),

            "DI_PLUS": float(
                latest["DI_PLUS"]
            ),

            "DI_MINUS": float(
                latest["DI_MINUS"]
            ),

            "volume_ratio": float(
                latest["VOLUME_RATIO"]
            )
        }
    }

    return result

def get_stock_price(ticker: str) -> dict:
    df = get_stock_data(ticker)

    latest = df.iloc[-1]

    return {
        "ticker": ticker,
        "open": float(latest["Open"]),
        "high": float(latest["High"]),
        "low": float(latest["Low"]),
        "close": float(latest["Close"]),
        "volume": int(latest["Volume"]),
    }

def get_technical_analysis(ticker: str) -> dict:
    df = get_stock_data(ticker)

    df = calculate_indicators(df)

    analysis = analyze_technical(df)

    score = calculate_technical_score(df)

    signal = make_trading_signal(score)

    latest = df.iloc[-1]

    return {
        "ticker": ticker,
        "price": float(latest["Close"]),
        "score": score,
        "signal": signal,
        "analysis": analysis,
        "indicators": {
            "MA5": float(latest["MA5"]),
            "MA20": float(latest["MA20"]),
            "MA60": float(latest["MA60"]),
            "RSI": float(latest["RSI"]),
            "MACD": float(latest["MACD"]),
            "MACD_SIGNAL": float(latest["MACD_SIGNAL"]),
            "ATR": float(latest["ATR"]),
            "ADX": float(latest["ADX"]),
            "DI_PLUS": float(latest["DI_PLUS"]),
            "DI_MINUS": float(latest["DI_MINUS"]),
            "VOLUME_RATIO": float(latest["VOLUME_RATIO"]),
        }
    }

#-------------------------------------------------
# 손절가 = 현재가 - 2 × ATR
# 목표가 = 현재가 + 3 × ATR
# -------------------------------------------------

def calculate_risk(
    ticker: str,
    account_size: float = 10000000,
    risk_per_trade: float = 0.01
) -> dict:

    df = get_stock_data(ticker)

    df = calculate_indicators(df)

    latest = df.iloc[-1]

    price = float(latest["Close"])
    atr = float(latest["ATR"])

    if not np.isfinite(price) or not np.isfinite(atr) or atr <= 0:
        raise ValueError(
            f"Insufficient price history to calculate risk for {ticker}"
        )

    stop_loss = price - (2 * atr)
    take_profit = price + (3 * atr)

    risk_amount = account_size * risk_per_trade

    risk_per_share = price - stop_loss

    if risk_per_share <= 0:
        position_size = 0
    else:
        position_size = int(
            risk_amount / risk_per_share
        )

    return {
        "ticker": ticker,
        "price": price,
        "ATR": atr,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "account_size": account_size,
        "risk_per_trade": risk_per_trade,
        "risk_amount": risk_amount,
        "position_size": position_size,
    }

