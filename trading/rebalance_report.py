from __future__ import annotations

from pathlib import Path

from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, Spacer

from trading.pdf_report import para, pdf_document, report_table, styles


def _money(value) -> str:
    try:
        return f"{float(value):,.0f}원"
    except (TypeError, ValueError):
        return "-"


def _pct(value, *, ratio=False) -> str:
    try:
        number = float(value) * (100 if ratio else 1)
        return f"{number:.1f}%"
    except (TypeError, ValueError):
        return "-"


def render_rebalance_report(package: dict, directory: Path) -> Path:
    """Create a Korean PDF proposal in the configured OneDrive sync folder."""
    trade_date = str(package["created_at"])[:10].replace("-", "")
    path = directory / f"KR Rebalancing_Proposal_{trade_date}.pdf"
    proposal = package["proposal"]
    snapshot = package["snapshot"]
    validation = package["validation"]
    style = styles()
    story = [
        para("LLM 포트폴리오 리밸런싱 제안", style["title"]),
        para(f"{package['proposal_id']} · {package['created_at']}", style["body"]),
        para("1. Executive Summary", style["heading"]),
        para(proposal.get("market_summary"), style["body"]),
        para(proposal.get("overall_reason"), style["body"]),
        Spacer(1, 3 * mm),
        report_table([
            [para(x, style["body"]) for x in ["시장 판단", "권장 현금", "총자산"]],
            [para(proposal.get("market_view"), style["body"]), para(_pct(proposal.get("recommended_cash_pct")), style["body"]), para(_money(snapshot.get("portfolio", {}).get("total_equity")), style["body"])],
        ], [55*mm, 45*mm, 55*mm], font_size=8),
        para("2. 포트폴리오 진단", style["heading"]),
        para(proposal.get("portfolio_assessment"), style["body"]),
    ]
    position_rows = [[para(x, style["small"]) for x in ["종목코드", "종목명", "수량", "평균단가", "현재가", "수익률", "비중"]]]
    for item in snapshot.get("positions", []):
        position_rows.append([para(x, style["small"]) for x in [
            item.get("ticker"), item.get("name"), f"{int(item.get('quantity', 0)):,}",
            _money(item.get("avg_price")), _money(item.get("current_price")),
            _pct(item.get("return_pct")), _pct(item.get("weight_pct")),
        ]])
    story.append(report_table(position_rows, [25*mm, 30*mm, 16*mm, 27*mm, 27*mm, 20*mm, 18*mm], font_size=7))
    story.extend([
        para("3. 시장 뉴스와 거시환경 (최근 7일)", style["heading"]),
        para(proposal.get("news_assessment"), style["body"]),
    ])
    for item in snapshot.get("market_news", {}).get("headlines", [])[:12]:
        story.append(para(f"• {item.get('title', '-')}", style["body"]))

    story.extend([PageBreak(), para("4. Top10 후보 비교", style["heading"])])
    top_rows = [[para(x, style["small"]) for x in ["순위", "종목명", "점수", "추천 근거"]]]
    for item in snapshot.get("top10", []):
        top_rows.append([para(x, style["small"]) for x in [
            item.get("rank"), item.get("name"), f"{float(item.get('total_score', 0)):.1f}",
            item.get("recommendation_reason"),
        ]])
    story.append(report_table(top_rows, [15*mm, 35*mm, 20*mm, 93*mm], font_size=7))
    story.extend([
        para("5. 리밸런싱 제안과 상세 근거", style["heading"]),
        para(proposal.get("investment_thesis"), style["body"]),
    ])
    security_names = {
        str(item.get("ticker") or "").upper(): item.get("name")
        for item in snapshot.get("positions", []) + snapshot.get("top10", [])
    }
    action_rows = [[para(x, style["small"]) for x in ["행동", "종목명", "목표비중", "신뢰도", "상세 근거"]]]
    for item in proposal.get("actions", []):
        name = security_names.get(str(item.get("ticker") or "").upper()) or item.get("ticker")
        detail = (
            f"{item.get('reason', '-')}\n지지: {', '.join(item.get('supporting_factors', [])) or '-'}"
            f"\n위험: {', '.join(item.get('risks', [])) or '-'}"
        )
        action_rows.append([para(x, style["small"]) for x in [
            item.get("action"), name, _pct(item.get("target_weight_pct")),
            _pct(item.get("confidence"), ratio=True), detail,
        ]])
    story.append(report_table(action_rows, [18*mm, 30*mm, 22*mm, 20*mm, 73*mm], font_size=7))

    story.extend([PageBreak(), para("6. 위험 시나리오", style["heading"])])
    for item in proposal.get("risk_scenarios", []) or ["-"]:
        story.append(para(f"• {item}", style["body"]))
    story.append(para("7. Risk Validator", style["heading"]))
    for item in validation.get("errors", []) or ["정책 위반 없음"]:
        story.append(para(f"• {item}", style["body"]))
    story.append(para("8. 실행 및 사후관리", style["heading"]))
    for item in proposal.get("implementation_notes", []) or ["-"]:
        story.append(para(f"• {item}", style["body"]))
    story.extend([Spacer(1, 5 * mm), para("자동 생성된 투자 검토 보조자료이며 수익을 보장하지 않습니다. 실제 주문은 사용자 승인과 Risk Validator 절차를 거칩니다.", style["small"])])
    pdf_document(path).build(story)
    return path
