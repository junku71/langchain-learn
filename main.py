import sys

from llm_agent import run_agent


sys.stdout.reconfigure(encoding="utf-8")

question = """
삼성전자 매수한다고 가정할 때
손절가와 목표가를 알려줘.
내 계좌는 5천만원이고
한 번 거래에서 최대 1%까지만 손실을 감수할게.
"""

answer = run_agent(
    question
)

print("\nFinal Answer:")
print(answer)
