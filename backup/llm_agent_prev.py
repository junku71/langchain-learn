import json

from dotenv import load_dotenv
from openai import OpenAI

from stock_analyzer import analyze_stock


load_dotenv()

client = OpenAI()


# ---------------------------------------
# LLM에게 알려줄 Tool 정의
# ---------------------------------------

tools = [
    {
        "type": "function",
        "name": "analyze_stock",
        "description": (
            "Analyze a stock using technical indicators "
            "including moving averages, RSI, MACD, ATR, "
            "ADX, DI+, DI-, and volume."
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
    }
]



def ask_stock_agent(user_message: str):

    response = client.responses.create(
        model="gpt-5.6",
        input=user_message,
        tools=tools
    )

    return response

# ---------------------------------------
# LLM에게 알려줄 Tool 실행 함수 정의
# ---------------------------------------
def execute_tool(
    tool_name: str,
    arguments: dict
):

    if tool_name == "analyze_stock":

        ticker = arguments["ticker"]

        return analyze_stock(
            ticker
        )

    raise ValueError(
        f"Unknown tool: {tool_name}"
    )

# ---------------------------------------
# Agent 비슷한 구조 
# ---------------------------------------

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

            arguments = json.loads(
                item.arguments
            )

            result = execute_tool(
                item.name,
                arguments
            )

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
