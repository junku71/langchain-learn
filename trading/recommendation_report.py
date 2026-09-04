from __future__ import annotations

from pathlib import Path

from reportlab.lib.units import mm
from reportlab.platypus import Spacer

from trading.pdf_report import para, pdf_document, report_table, styles


def _score(value) -> str:
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return "-"


def render_top10_pick_report(
    recommendations: list[dict], *, trade_date: str,
    universe_scope: str, directory: Path,
) -> Path:
    """Create the Top10 PDF directly in the configured OneDrive folder."""
    path = directory / (
        f"KR_Top10_Pick_{trade_date.replace('-', '')}_{universe_scope.upper()}.pdf"
    )
    style = styles()
    story = [
        para("오늘의 Korea Top10 pick", style["title"]),
        para(
            f"분석일 {trade_date} · 유니버스 {universe_scope.upper()} · "
            f"총 {len(recommendations)}종목", style["body"],
        ),
        Spacer(1, 5 * mm),
    ]
    header = ["순위", "종목명", "시장·섹터", "종합", "기본 분석", "기술 분석", "수급 분석", "뉴스 분석", "핵심 추천근거"]
    rows = [[para(x, style["small"]) for x in header]]
    for item in recommendations:
        rows.append([
            para(item.get("rank"), style["small"]),
            para(f"{item.get('name', '-')}\n{item.get('ticker', '-')}", style["small"]),
            para(f"{item.get('market', '-')}\n{item.get('sector', '-')}", style["small"]),
            para(_score(item.get("total_score")), style["small"]),
            para(f"{_score(item.get('fundamental_score'))}\n{item.get('fundamental_reason', '-')}", style["small"]),
            para(f"{_score(item.get('technical_score'))}\n{item.get('technical_reason', '-')}", style["small"]),
            para(f"{_score(item.get('flow_score'))}\n{item.get('flow_reason', '-')}", style["small"]),
            para(f"{_score(item.get('news_score'))}\n{item.get('news_reason', '-')}", style["small"]),
            para(item.get("recommendation_reason"), style["small"]),
        ])
    story.append(report_table(rows, [10*mm, 25*mm, 22*mm, 13*mm, 39*mm, 39*mm, 39*mm, 39*mm, 42*mm], font_size=6.2))
    story.extend([Spacer(1, 5 * mm), para("자동 생성된 투자 검토 보조자료이며 수익을 보장하지 않습니다.", style["small"])])
    pdf_document(path, landscape_page=True).build(story)
    return path
