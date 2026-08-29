from typing import TypedDict
import pandas as pd
from dotenv import load_dotenv


from typing import TypedDict



from ml.ml_filter import (
    predict_up_probability
)

from risk.risk_config import (
    RiskConfig,
)

from risk.risk_engine import (
    calculate_position_risk,
)

load_dotenv()

#  Technical Agent
#  → technical_result
#  
#  Fundamental Agent
#  → fundamental_result
#  
#  News Agent
#  → news_result
#  
#  Flow Agent
#  → flow_result

class StockState(TypedDict):

    ticker: str
    account_size: float
    market_data: pd.DataFrame | None

    technical_result: dict | None

    fundamental_result: dict | None

    news_result: dict | None

    flow_result: dict | None

    merged_result: dict | None

    ml_result: dict | None
    risk_result: dict | None

    final_decision: str | None



#------------------------------------
# technical agent용 tool
#------------------------------------
from langchain_core.tools import tool

from stock_analyzer import (
    calculate_indicators,
    get_stock_data,
    get_technical_analysis,
)

#------------------------------------
# route_after_risk
#-----------------------------------

def route_after_risk(
    state: StockState
):

    approved = state[
        "risk_result"
    ][
        "approved"
    ]

    if approved:
        return "approve"

    return "reject"

#------------------------------------
# route_after_ml
#-----------------------------------

def route_after_ml(
    state: StockState
):

    probability = state[
        "ml_result"
    ][
        "up_probability"
    ]

    if probability >= 0.65:

        return "pass"

    return "reject"




@tool
def technical_analysis_tool(
    ticker: str
) -> dict:
    """
    Perform technical analysis of a stock.

    Use RSI, MACD, moving averages, ATR,
    ADX, DI+, DI-, and volume.

    Use this tool when evaluating
    trend, momentum, and technical signal.
    """

    return get_technical_analysis(
        ticker
    )

# technical agent 만들기
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

llm = ChatOpenAI(
    model="gpt-5.6",
    temperature=0,
    use_responses_api=True,
)

technical_agent = create_agent(
    model=llm,

    tools=[
        technical_analysis_tool
    ],

    system_prompt="""
    You are a technical stock analysis agent.

    Your job is to evaluate:
    - trend
    - momentum
    - volatility
    - RSI
    - MACD
    - ADX
    - DI+
    - DI-
    - volume

    Always use the technical analysis tool.

    Return a concise conclusion
    with a score from 0 to 100
    and one of:
    BULLISH, NEUTRAL, BEARISH.
    """
)


# fundamental agent 만들기 ( 학습용 가짜 에이전트 )

@tool
def fundamental_analysis_tool(
    ticker: str
) -> dict:
    """
    Return fundamental indicators
    for a stock.
    """

    return {
        "ticker": ticker,
        "PER": 14.5,
        "PBR": 1.3,
        "ROE": 12.8,
        "debt_ratio": 35.2,
    }

fundamental_agent = create_agent(
    model=llm,

    tools=[
        fundamental_analysis_tool
    ],

    system_prompt="""
    You are a fundamental stock analyst.

    Evaluate valuation and financial quality.

    Consider:
    - PER
    - PBR
    - ROE
    - debt ratio

    Always use the fundamental analysis tool.

    Return:
    - score from 0 to 100
    - BULLISH, NEUTRAL, or BEARISH
    - short reasoning
    """
)


# news agent 만들기 ( 학습용 가짜 에이전트 )

@tool
def news_analysis_tool(
    ticker: str
) -> dict:
    """
    Return recent news sentiment
    information for the stock.
    """

    return {
        "ticker": ticker,
        "positive_news": 7,
        "negative_news": 2,
        "sentiment_score": 72,
    }

news_agent = create_agent(
    model=llm,

    tools=[
        news_analysis_tool
    ],

    system_prompt="""
    You are a stock news sentiment analyst.

    Evaluate recent company and market news.

    Always use the news analysis tool.

    Return:
    - sentiment score from 0 to 100
    - POSITIVE, NEUTRAL, or NEGATIVE
    - short reasoning
    """
)

# flow agent 만들기 ( 학습용 가짜 에이전트 )
@tool
def flow_analysis_tool(
    ticker: str
) -> dict:
    """
    Analyze foreign and institutional
    investor trading flows.
    """

    return {
        "ticker": ticker,
        "foreign_net_buy": 1200000,
        "institution_net_buy": 450000,
        "foreign_buy_days": 4,
        "institution_buy_days": 3,
    }

flow_agent = create_agent(
    model=llm,

    tools=[
        flow_analysis_tool
    ],

    system_prompt="""
    You are an investor flow analysis agent.
    
    Evaluate:
    - foreign investor buying
    - institutional buying
    - consecutive net-buying days
    
    Always use the flow analysis tool.
    
    Return:
    - score from 0 to 100
    - BULLISH, NEUTRAL, or BEARISH
    - short reasoning
    """
)

# 의사결정 에이전트 만들기 

decision_agent = create_agent(
    model=llm,

    tools=[],

    system_prompt="""
    You are the head stock decision agent.

    You receive analysis from:
    1. Technical Agent
    2. Fundamental Agent
    3. News Agent
    4. Flow Agent

    Evaluate all evidence.

    Return exactly one decision:

    BUY
    HOLD
    SELL

    Also return a short reason.
    """
)

#------------------------------------
# 시장 데이터를 가져오는 노드
def market_data_node(
    state: StockState
):

    ticker = state["ticker"]

    df = get_stock_data(
        ticker
    )

    return {
        "market_data": df
    }


def technical_agent_node(
    state: StockState
):

    ticker = state["ticker"]

    result = technical_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content":
                        f"Analyze {ticker}"
                }
            ]
        }
    )

    return {
        "technical_result": result["messages"][-1].text
    }

#------------------------------------
# fundamental 노드: 학습용 가짜 함수
#------------------------------------

def fundamental_agent_node(
    state: StockState
):

    ticker = state["ticker"]

    result = fundamental_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content":
                        f"Analyze {ticker}"
                }
            ]
        }
    )

    return {
        "fundamental_result": result["messages"][-1].text
    }

#------------------------------------
# 외국인/기관 수습분석 노드: 학습용 가짜 함수
#------------------------------------
def flow_agent_node(
    state: StockState
):

    ticker = state["ticker"]

    result = flow_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content":
                        f"Analyze investor flow for {ticker}"
                }
            ]
        }
    )

    return {
        "flow_result": result["messages"][-1].text
    }


#------------------------------------
# news 노드: 학습용 가짜 함수
#------------------------------------
def news_agent_node(
    state: StockState
):

    ticker = state["ticker"]

    result = news_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content":
                        f"Analyze recent news for {ticker}"
                }
            ]
        }
    )

    return {
        "news_result": result["messages"][-1].text
    }

#------------------------------------
# merge 노드: 기술적, 기본적, 뉴스, 외국인/기관 수
#------------------------------------

def merge_node(
    state: StockState
):

    merged = {
        "technical":
            state["technical_result"],

        "fundamental":
            state["fundamental_result"],

        "news":
            state["news_result"],

        "flow":
            state["flow_result"],
    }

    return {
        "merged_result": merged
    }


#------------------------------------
# ML Filter 노드
#------------------------------------
def ml_filter_node(
    state: StockState
):

    print(
        "[ML Filter 시작]"
    )

    df = state[
        "market_data"
    ]

    result = (
        predict_up_probability(
            df
        )
    )

    print(
        "[ML Result]",
        result
    )

    return {
        "ml_result": result
    }

#------------------------------------
# MLFilter결과에 의한 Reject 노드
#------------------------------------
def reject_node(
    state: StockState
):

    probability = state[
        "ml_result"
    ][
        "up_probability"
    ]

    return {
        "final_decision": (
            f"HOLD - "
            f"ML probability "
            f"{probability:.2%}"
        )
    }

#------------------------------------
# Risk 노드: 포지션 위험 계산
#------------------------------------

def risk_node(
    state: StockState
):

    df = calculate_indicators(state[
        "market_data"
    ])

    latest = df.iloc[-1]

    price = float(
        latest["Close"]
    )

    atr = float(
        latest["ATR"]
    )

    config = RiskConfig()

    result = calculate_position_risk(
        price=price,
        atr=atr,
        account_size=state[
            "account_size"
        ],
        config=config,
    )

    return {
        "risk_result": result
    }

#------------------------------------
# Decision node: 매수, 매도, 관망 결정
#------------------------------------


def decision_node(
    state: StockState
):

    merged = state[
        "merged_result"
    ]

    result = decision_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content":
                        f"""
                        Analyze the following results
                        and make the final decision.

                        {merged}
                        """
                }
            ]
        }
    )

    return {
        "final_decision": result["messages"][-1].text
    }

#------------------------------------
# Risk Reject 노드
#------------------------------------

def risk_reject_node(
    state: StockState
):

    return {
        "final_decision":
            "HOLD - Risk rejected"
    }

#------------------------------------
# 그래프를 만듬
#------------------------------------


from langgraph.graph import (
    StateGraph,
    START,
    END,
)


builder = StateGraph(
    StockState
)


builder.add_node(
    "market_data",
    market_data_node
)

builder.add_node(
    "technical",
    technical_agent_node
)

builder.add_node(
    "fundamental",
    fundamental_agent_node
)

builder.add_node(
    "news",
    news_agent_node
)

builder.add_node(
    "flow",
    flow_agent_node
)

builder.add_node(
    "merge",
    merge_node
)

builder.add_node(
    "ml_filter",
    ml_filter_node
)

builder.add_node(
    "reject",
    reject_node
)


builder.add_node(
    "risk_reject",
    risk_reject_node
)

builder.add_node(
    "risk",
    risk_node
)

builder.add_node(
    "decision",
    decision_node
)


#------------------------------------
# Noedes를 연결함
#------------------------------------


builder.add_edge(
    START,
    "market_data"
)

builder.add_edge(
    "market_data",
    "technical"
)

builder.add_edge(
    "market_data",
    "fundamental"
)

builder.add_edge(
    "market_data",
    "news"
)

builder.add_edge(
    "market_data",
    "flow"
)

# -------------------------
# Fan In
# -------------------------

builder.add_edge(
    [
        "technical",
        "fundamental",
        "news",
        "flow",
    ],
    "merge"
)

builder.add_edge(
    "merge",
    "ml_filter"
)

builder.add_conditional_edges(
    "ml_filter",
    route_after_ml,
    {
        "pass": "risk",
        "reject": "reject",
    }
)

builder.add_conditional_edges(
    "risk",
    route_after_risk,
    {
        "approve":
            "decision",

        "reject":
            "risk_reject",
    }
)

builder.add_edge(
    "reject",
    END
)

builder.add_edge(
    "risk_reject",
    END
)

#------------------------------------
# 그래프를 컴파일함
#   Graph 설계
#   ↓
#   compile
#   ↓
#   invoke
#------------------------------------

graph = builder.compile()
