from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from dotenv import load_dotenv

from stock_analyzer import (
    get_stock_price,
    get_technical_analysis,
    calculate_risk,
)


load_dotenv()


# ------------------------------------
# Tool 1
# ------------------------------------

@tool
def stock_price_tool(
    ticker: str
) -> dict:

    """
    Get the latest stock price
    and OHLCV data.

    ticker:
    Yahoo Finance ticker symbol.

    Example:
    Samsung Electronics = 005930.KS
    """

    return get_stock_price(
        ticker
    )


# ------------------------------------
# Tool 2
# ------------------------------------

@tool
def technical_analysis_tool(
    ticker: str
) -> dict:

    """
    Perform technical analysis.

    Uses:
    moving averages,
    RSI,
    MACD,
    ATR,
    ADX,
    DI+,
    DI-,
    volume.
    """

    return get_technical_analysis(
        ticker
    )


# ------------------------------------
# Tool 3
# ------------------------------------

@tool
def risk_analysis_tool(
    ticker: str,
    account_size: float = 10000000,
    risk_per_trade: float = 0.01
) -> dict:

    """
    Calculate ATR based trading risk.

    Returns:
    stop loss,
    take profit,
    risk amount,
    position size.
    """

    return calculate_risk(
        ticker=ticker,
        account_size=account_size,
        risk_per_trade=risk_per_trade,
    )


# ------------------------------------
# Tool List
# ------------------------------------

tools = [
    stock_price_tool,
    technical_analysis_tool,
    risk_analysis_tool,
]


# ------------------------------------
# LLM
# ------------------------------------

llm = ChatOpenAI(
    model="gpt-5.6",
    temperature=0,
    use_responses_api=True,
)


# ------------------------------------
# Agent
# ------------------------------------

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        "답변은 한국어 일반 텍스트로 작성하세요. "
        "터미널에서 읽기 쉽도록 짧은 제목과 목록을 사용하되, "
        "Markdown 표, # 제목 기호, 굵은 글씨 기호, LaTeX 수식은 사용하지 마세요. "
        "가격과 위험 계산의 핵심 결론을 명확하게 요약하세요."
    ),
)


# ------------------------------------
# Run
# ------------------------------------

def run_stock_agent(
    question: str
) -> str:

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question,
                }
            ]
        }
    )

    if not result["messages"]:
        return "분석 결과를 생성하지 못했습니다."

    return result["messages"][-1].text
