import sys

from multiagent_graph import graph


sys.stdout.reconfigure(encoding="utf-8")

initial_state = {

    "ticker":
        "005930.KS",

    "account_size":
        50000000,

    "market_data":
        None,

    "technical_result":
        None,

    "fundamental_result":
        None,

    "news_result":
        None,

    "flow_result":
        None,

    "merged_result":
        None,

    "ml_result":
        None,

    "risk_result":
        None,

    "final_decision":
        None,
}


for event in graph.stream(
    initial_state
):

    print(
        "\n===================="
    )

    print(event)
