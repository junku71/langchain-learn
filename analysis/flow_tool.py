from langchain_core.tools import tool

from analysis.flow import analyze_flow


@tool
def flow_analysis_tool(ticker: str) -> dict:
    """Analyze foreign and institutional investor flows for a Korean stock.

    Evaluates cumulative net buying, simultaneous buying days, and buying
    persistence. Returns a 0-100 score and BULLISH/NEUTRAL/BEARISH signal.
    """
    return analyze_flow(ticker=ticker, lookback=20)
