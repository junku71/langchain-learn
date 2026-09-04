from dataclasses import dataclass


#  risk_per_trade
#  → 거래당 계좌 위험 1%
#  
#  atr_stop_multiple
#  → ATR × 2 손절
#  
#  reward_risk_ratio
#  → Risk 대비 목표 수익 배수
#  
#  max_position_pct
#  → 한 종목 최대 20%
#  
#  max_portfolio_risk_pct
#  → 전체 포트폴리오 최대 위험 5%

@dataclass
class RiskConfig:

    risk_per_trade: float = 0.01

    atr_stop_multiple: float = 2.0

    reward_risk_ratio: float = 1.5

    max_position_pct: float = 0.20

    max_portfolio_risk_pct: float = 0.05