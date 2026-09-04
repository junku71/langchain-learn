import json

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from analysis.technical import (
    get_stock_price,
    get_technical_analysis,
    calculate_risk,
)


client = OpenAI()

tools = [
    {
        "type": "function",
        "name": "get_stock_price",
        "description": (
            "Get the latest stock price and OHLCV data."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": (
                        "Yahoo Finance ticker symbol. "
                        "Example: Samsung Electronics = 005930.KS"
                    )
                }
            },
            "required": ["ticker"],
            "additionalProperties": False
        }
    },

    {
        "type": "function",
        "name": "get_technical_analysis",
        "description": (
            "Analyze a stock using technical indicators "
            "such as moving averages, RSI, MACD, ATR, ADX, "
            "DI+, DI-, and volume."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string"
                }
            },
            "required": ["ticker"],
            "additionalProperties": False
        }
    },

    {
        "type": "function",
        "name": "calculate_risk",
        "description": (
            "Calculate ATR-based stop loss, take profit, "
            "risk amount, and position size."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string"
                },

                "account_size": {
                    "type": "number",
                    "description": (
                        "Total investment account size in KRW."
                    )
                },

                "risk_per_trade": {
                    "type": "number",
                    "description": (
                        "Maximum fraction of the account to risk "
                        "on one trade. Example: 0.01 = 1%."
                    )
                }
            },
            "required": ["ticker"],
            "additionalProperties": False
        }
    }
]


def execute_tool(
    tool_name: str,
    arguments: dict
):

    if tool_name == "get_stock_price":

        return get_stock_price(
            arguments["ticker"]
        )


    elif tool_name == "get_technical_analysis":

        return get_technical_analysis(
            arguments["ticker"]
        )


    elif tool_name == "calculate_risk":

        return calculate_risk(
            ticker=arguments["ticker"],
            account_size=arguments.get(
                "account_size",
                10000000
            ),
            risk_per_trade=arguments.get(
                "risk_per_trade",
                0.01
            )
        )


    else:

        raise ValueError(
            f"Unknown tool: {tool_name}"
        )


def run_agent(
    user_message: str
):

    response = client.responses.create(
        model="gpt-5.6",
        input=user_message,
        tools=tools
    )

    tool_outputs = []

    for item in response.output:

        if item.type == "function_call":

            tool_name = item.name

            arguments = json.loads(
                item.arguments
            )

            print("\n---------------------")
            print("Selected Tool")
            print("---------------------")

            print(tool_name)

            print("\nArguments:")

            print(arguments)


            result = execute_tool(
                tool_name,
                arguments
            )

            print("\nTool Result:")

            print(result)


            tool_outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": json.dumps(
                        result,
                        ensure_ascii=False
                    )
                }
            )


    if tool_outputs:

        final_response = client.responses.create(
            model="gpt-5.6",

            previous_response_id=response.id,

            input=tool_outputs,

            tools=tools
        )

        return final_response.output_text


    return response.output_text
